"""
Views for DataLens.
"""
import os
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages

from .models import Dataset, ChatMessage
from . import data_processor, visualizations, llm, reports


# In-memory cache for dataframes — keyed by dataset id.
# Avoids re-reading from disk on every request.
_DF_CACHE = {}


def get_dataframe(dataset):
    """Get or load the dataframe for a tabular dataset."""
    if dataset.file_type != 'tabular':
        return None
    if str(dataset.id) in _DF_CACHE:
        return _DF_CACHE[str(dataset.id)]
    df = data_processor.load_tabular(dataset.file.path)
    _DF_CACHE[str(dataset.id)] = df
    return df


def set_dataframe(dataset, df):
    """Replace the cached dataframe (e.g. after cleaning)."""
    _DF_CACHE[str(dataset.id)] = df


# ----------------- Pages -----------------

def home(request):
    """Landing page — list datasets + upload."""
    datasets = Dataset.objects.all()[:20]
    return render(request, 'home.html', {'datasets': datasets})


def dataset_view(request, dataset_id):
    """Main dataset view — stats, charts, chat."""
    dataset = get_object_or_404(Dataset, id=dataset_id)

    context = {'dataset': dataset}

    if dataset.file_type == 'tabular':
        try:
            df = get_dataframe(dataset)
            summary = data_processor.analyze_tabular(df)
            preview = data_processor.get_preview(df, n=20)
            chart_options = visualizations.get_chart_options(df)

            context.update({
                'summary': summary,
                'preview': preview,
                'chart_options': json.dumps(chart_options),
                'column_info_json': json.dumps(summary['column_info']),
            })
        except Exception as e:
            messages.error(request, f"Error loading dataset: {e}")

    elif dataset.file_type == 'image':
        info = data_processor.analyze_image(dataset.file.path)
        try:
            data_url = data_processor.image_to_base64(dataset.file.path)
        except Exception:
            data_url = None
        context.update({'image_info': info, 'image_data_url': data_url})

    elif dataset.file_type == 'pdf':
        info = data_processor.analyze_pdf(dataset.file.path)
        context['pdf_info'] = info

    elif dataset.file_type == 'text':
        info = data_processor.analyze_text(dataset.file.path)
        context['text_info'] = info

    # Chat history
    context['chat_messages'] = list(dataset.messages.all())

    return render(request, 'dataset.html', context)


# ----------------- Upload -----------------

@require_POST
def upload_file(request):
    """Handle file upload."""
    if 'file' not in request.FILES:
        messages.error(request, "No file provided")
        return redirect('home')

    f = request.FILES['file']
    file_type = data_processor.detect_file_type(f.name)

    dataset = Dataset.objects.create(
        original_filename=f.name,
        file=f,
        file_type=file_type,
        file_size=f.size,
    )

    # Cache metadata for tabular files
    if file_type == 'tabular':
        try:
            df = data_processor.load_tabular(dataset.file.path)
            _DF_CACHE[str(dataset.id)] = df
            dataset.rows = len(df)
            dataset.columns = len(df.columns)
            dataset.save()
        except Exception as e:
            messages.warning(request, f"Uploaded, but could not parse: {e}")

    return redirect('dataset_view', dataset_id=dataset.id)


@require_POST
def delete_dataset(request, dataset_id):
    """Delete a dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    if str(dataset.id) in _DF_CACHE:
        del _DF_CACHE[str(dataset.id)]
    try:
        if dataset.file and os.path.exists(dataset.file.path):
            os.remove(dataset.file.path)
    except Exception:
        pass
    dataset.delete()
    return redirect('home')


# ----------------- Chart API -----------------

@require_POST
def generate_chart(request, dataset_id):
    """Generate a chart based on user selection."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    if dataset.file_type != 'tabular':
        return JsonResponse({'error': 'Charts only work for tabular data'}, status=400)

    try:
        body = json.loads(request.body)
        chart_type = body.get('chart_type')
        df = get_dataframe(dataset)

        if chart_type == 'histogram':
            result = visualizations.histogram(df, body['x_col'], bins=body.get('bins', 30))
        elif chart_type == 'bar':
            result = visualizations.bar_chart(
                df, body['x_col'], body.get('y_col'), body.get('agg', 'count')
            )
        elif chart_type == 'line':
            result = visualizations.line_chart(df, body['x_col'], body['y_col'])
        elif chart_type == 'scatter':
            result = visualizations.scatter_plot(
                df, body['x_col'], body['y_col'], body.get('color_col')
            )
        elif chart_type == 'box':
            result = visualizations.box_plot(df, body['x_col'], body.get('group_by'))
        elif chart_type == 'pie':
            result = visualizations.pie_chart(df, body['x_col'])
        elif chart_type == 'correlation':
            result = visualizations.correlation_heatmap(df)
        else:
            return JsonResponse({'error': f'Unknown chart type: {chart_type}'}, status=400)

        return JsonResponse({'chart': result})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# ----------------- Cleaning API -----------------

@require_POST
def clean_dataset(request, dataset_id):
    """Apply cleaning operations to the dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    if dataset.file_type != 'tabular':
        return JsonResponse({'error': 'Cleaning only works for tabular data'}, status=400)

    try:
        operations = json.loads(request.body)
        df = get_dataframe(dataset)
        cleaned, log = data_processor.clean_data(df, operations)
        set_dataframe(dataset, cleaned)

        # Update cached metadata
        dataset.rows = len(cleaned)
        dataset.columns = len(cleaned.columns)
        dataset.save()

        summary = data_processor.analyze_tabular(cleaned)
        preview = data_processor.get_preview(cleaned, n=20)

        return JsonResponse({
            'log': log,
            'summary': summary,
            'preview': preview,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# ----------------- LLM Chat API -----------------

@require_POST
def chat_message(request, dataset_id):
    """Send a chat message about the dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    body = json.loads(request.body)
    user_message = body.get('message', '').strip()
    if not user_message:
        return JsonResponse({'error': 'Empty message'}, status=400)

    # Build context based on file type
    extra_context = None
    summary = None
    if dataset.file_type == 'tabular':
        try:
            df = get_dataframe(dataset)
            summary = data_processor.analyze_tabular(df)
        except Exception as e:
            return JsonResponse({'error': f"Could not load data: {e}"}, status=500)
    elif dataset.file_type == 'image':
        extra_context = data_processor.analyze_image(dataset.file.path)
    elif dataset.file_type == 'pdf':
        extra_context = data_processor.analyze_pdf(dataset.file.path)
    elif dataset.file_type == 'text':
        extra_context = data_processor.analyze_text(dataset.file.path)

    data_context = llm.build_data_context(dataset, summary=summary, extra_context=extra_context)

    # Save user message
    ChatMessage.objects.create(dataset=dataset, role='user', content=user_message)

    # Get assistant response
    history = list(dataset.messages.all().exclude(content=user_message).order_by('created_at'))
    response_text = llm.chat(dataset, user_message, history[:-1] if history else [], data_context)

    ChatMessage.objects.create(dataset=dataset, role='assistant', content=response_text)

    return JsonResponse({'reply': response_text})


@require_POST
def clear_chat(request, dataset_id):
    """Clear chat history for a dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    dataset.messages.all().delete()
    return JsonResponse({'ok': True})


# ----------------- Report Export -----------------

def export_pdf(request, dataset_id):
    """Generate and download PDF report."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    if dataset.file_type != 'tabular':
        return HttpResponse("PDF reports only available for tabular data", status=400)

    try:
        df = get_dataframe(dataset)
        summary = data_processor.analyze_tabular(df)
        preview = data_processor.get_preview(df, n=20)
        pdf_bytes = reports.generate_report(dataset, summary, preview)

        response = HttpResponse(pdf_bytes, content_type='application/pdf')
        safe_name = dataset.original_filename.rsplit('.', 1)[0]
        response['Content-Disposition'] = f'attachment; filename="{safe_name}_report.pdf"'
        return response
    except Exception as e:
        return HttpResponse(f"Error generating report: {e}", status=500)


def download_data(request, dataset_id):
    """Download the (possibly cleaned) data as CSV."""
    dataset = get_object_or_404(Dataset, id=dataset_id)
    if dataset.file_type != 'tabular':
        return HttpResponse("Only tabular data can be downloaded as CSV", status=400)

    df = get_dataframe(dataset)
    csv_data = df.to_csv(index=False)
    response = HttpResponse(csv_data, content_type='text/csv')
    safe_name = dataset.original_filename.rsplit('.', 1)[0]
    response['Content-Disposition'] = f'attachment; filename="{safe_name}_cleaned.csv"'
    return response
