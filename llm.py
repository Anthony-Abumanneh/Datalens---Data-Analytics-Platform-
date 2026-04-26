"""
LLM module — chat with your data using OpenAI.
"""
import os
from django.conf import settings


SYSTEM_PROMPT = """You are DataLens AI, an expert data analyst assistant. You help users understand and interpret their data.

You will be given context about a dataset (column names, statistics, sample values) and the user will ask questions about it.

Guidelines:
- Be concise but thorough
- Suggest specific charts or analyses when relevant ("Try a histogram of the 'price' column")
- If the user asks for something you can't directly do (like running custom code), explain what they should look at
- When discussing statistics, give plain-English interpretations
- Don't invent data. If the answer isn't in the context provided, say so
- Format numbers nicely (e.g., 1,234 not 1234)
- Be friendly and encouraging — many users are non-technical
"""


def get_client():
    """Get OpenAI client, or None if no API key."""
    api_key = settings.OPENAI_API_KEY
    if not api_key or api_key.startswith('sk-your'):
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def build_data_context(dataset, summary=None, extra_context=None):
    """Build a context string describing the dataset for the LLM."""
    parts = [f"Dataset: {dataset.original_filename} ({dataset.file_type})"]

    if dataset.file_type == 'tabular' and summary:
        parts.append(f"Rows: {summary['rows']:,}, Columns: {summary['columns']}")
        parts.append(f"Total nulls: {summary['total_nulls']:,} ({summary['null_pct_total']}%)")
        parts.append(f"Duplicate rows: {summary['duplicate_rows']:,}")
        parts.append("\nColumns:")
        for col in summary['column_info']:
            line = f"  - {col['name']} ({col['kind']}, dtype={col['dtype']}, "
            line += f"nulls={col['null_count']}, unique={col['unique']})"
            if col['kind'] == 'numeric' and 'mean' in col:
                line += f"  [min={col.get('min')}, max={col.get('max')}, mean={col.get('mean')}]"
            elif col['kind'] == 'categorical' and 'top_values' in col:
                top = ', '.join(f"{v['value']}({v['count']})" for v in col['top_values'][:3])
                line += f"  [top: {top}]"
            parts.append(line)

    elif dataset.file_type == 'image' and extra_context:
        parts.append(f"Image: {extra_context.get('width')}x{extra_context.get('height')} {extra_context.get('format')}")
        parts.append(f"Mode: {extra_context.get('mode')}, Aspect ratio: {extra_context.get('aspect_ratio')}")

    elif dataset.file_type == 'pdf' and extra_context:
        parts.append(f"PDF: {extra_context.get('num_pages')} pages, {extra_context.get('total_words'):,} words")
        if extra_context.get('preview'):
            parts.append(f"\nText content (first 3000 chars):\n{extra_context['preview']}")

    elif dataset.file_type == 'text' and extra_context:
        parts.append(f"Text file: {extra_context.get('words'):,} words, {extra_context.get('lines'):,} lines")
        if extra_context.get('preview'):
            parts.append(f"\nContent preview:\n{extra_context['preview']}")

    return '\n'.join(parts)


def chat(dataset, user_message, history, data_context, model='gpt-4o-mini'):
    """
    Send a chat message to OpenAI with dataset context.
    Returns the assistant's reply text.
    """
    client = get_client()
    if client is None:
        return ("⚠️ OpenAI API key is not configured. Add OPENAI_API_KEY to your "
                ".env file to enable AI chat.")

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'system', 'content': f"DATASET CONTEXT:\n{data_context}"},
    ]
    for msg in history:
        messages.append({'role': msg.role, 'content': msg.content})
    messages.append({'role': 'user', 'content': user_message})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error talking to OpenAI: {str(e)}"
