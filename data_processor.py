"""
Data processing module — loads, analyzes, and visualizes any file type.
"""
import os
import io
import json
import base64
import pandas as pd
import numpy as np
from PIL import Image
import pdfplumber


# ----------------- File type detection -----------------

TABULAR_EXTS = {'.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.json'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
PDF_EXTS = {'.pdf'}
TEXT_EXTS = {'.txt', '.md', '.log'}


def detect_file_type(filename):
    """Return one of: tabular, image, pdf, text, unknown."""
    ext = os.path.splitext(filename.lower())[1]
    if ext in TABULAR_EXTS:
        return 'tabular'
    if ext in IMAGE_EXTS:
        return 'image'
    if ext in PDF_EXTS:
        return 'pdf'
    if ext in TEXT_EXTS:
        return 'text'
    return 'unknown'


# ----------------- Loaders -----------------

def load_tabular(filepath):
    """Load any tabular file into a pandas DataFrame."""
    ext = os.path.splitext(filepath.lower())[1]
    try:
        if ext == '.csv':
            return pd.read_csv(filepath)
        if ext == '.tsv':
            return pd.read_csv(filepath, sep='\t')
        if ext in ('.xlsx', '.xls'):
            return pd.read_excel(filepath)
        if ext == '.parquet':
            return pd.read_parquet(filepath)
        if ext == '.json':
            try:
                return pd.read_json(filepath)
            except ValueError:
                # Try lines format
                return pd.read_json(filepath, lines=True)
    except Exception as e:
        raise ValueError(f"Could not parse file: {e}")
    raise ValueError(f"Unsupported tabular extension: {ext}")


# ----------------- Tabular analysis -----------------

def analyze_tabular(df):
    """Return summary statistics + column info for a DataFrame."""
    column_info = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        info = {
            'name': str(col),
            'dtype': dtype,
            'non_null': int(series.notna().sum()),
            'null_count': int(series.isna().sum()),
            'null_pct': round(float(series.isna().mean() * 100), 2),
            'unique': int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            info['kind'] = 'numeric'
            non_null = series.dropna()
            if len(non_null) > 0:
                info['min'] = _safe_float(non_null.min())
                info['max'] = _safe_float(non_null.max())
                info['mean'] = _safe_float(non_null.mean())
                info['median'] = _safe_float(non_null.median())
                info['std'] = _safe_float(non_null.std())
        elif pd.api.types.is_datetime64_any_dtype(series):
            info['kind'] = 'datetime'
            non_null = series.dropna()
            if len(non_null) > 0:
                info['min'] = str(non_null.min())
                info['max'] = str(non_null.max())
        else:
            info['kind'] = 'categorical'
            top = series.value_counts().head(5)
            info['top_values'] = [
                {'value': str(k), 'count': int(v)} for k, v in top.items()
            ]

        column_info.append(info)

    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'duplicate_rows': int(df.duplicated().sum()),
        'total_nulls': int(df.isna().sum().sum()),
        'null_pct_total': round(float(df.isna().mean().mean() * 100), 2),
        'column_info': column_info,
    }
    return summary


def _safe_float(v):
    """Convert to float, handling NaN/inf."""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def get_preview(df, n=10):
    """Get first N rows as a JSON-serializable dict."""
    preview = df.head(n).copy()
    # Handle dates and other non-serializable types
    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].astype(str)
    preview = preview.replace({np.nan: None, np.inf: None, -np.inf: None})
    return {
        'columns': [str(c) for c in preview.columns],
        'rows': preview.values.tolist(),
    }


# ----------------- Data cleaning ops -----------------

def clean_data(df, operations):
    """
    Apply cleaning operations.
    operations is a dict with keys like: drop_duplicates, drop_nulls,
    fill_nulls (with method), drop_columns (list)
    """
    result = df.copy()
    log = []

    if operations.get('drop_duplicates'):
        before = len(result)
        result = result.drop_duplicates()
        log.append(f"Removed {before - len(result)} duplicate rows")

    drop_cols = operations.get('drop_columns', [])
    if drop_cols:
        existing = [c for c in drop_cols if c in result.columns]
        result = result.drop(columns=existing)
        log.append(f"Dropped columns: {', '.join(existing)}")

    null_op = operations.get('null_handling')
    if null_op == 'drop_rows':
        before = len(result)
        result = result.dropna()
        log.append(f"Removed {before - len(result)} rows with nulls")
    elif null_op == 'fill_zero':
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)
        log.append(f"Filled numeric nulls with 0")
    elif null_op == 'fill_mean':
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col] = result[col].fillna(result[col].mean())
        log.append(f"Filled numeric nulls with column mean")
    elif null_op == 'fill_median':
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col] = result[col].fillna(result[col].median())
        log.append(f"Filled numeric nulls with column median")

    return result, log


# ----------------- Image analysis -----------------

def analyze_image(filepath):
    """Extract metadata from image file."""
    try:
        img = Image.open(filepath)
        return {
            'format': img.format,
            'mode': img.mode,
            'width': img.width,
            'height': img.height,
            'size_pixels': img.width * img.height,
            'aspect_ratio': round(img.width / img.height, 3) if img.height else 0,
        }
    except Exception as e:
        return {'error': str(e)}


def image_to_base64(filepath, max_size=1200):
    """Return image as data URL, scaled down if huge."""
    img = Image.open(filepath)
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    fmt = 'PNG' if img.mode in ('RGBA', 'LA', 'P') else 'JPEG'
    if fmt == 'JPEG' and img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode()
    mime = 'image/png' if fmt == 'PNG' else 'image/jpeg'
    return f"data:{mime};base64,{encoded}"


# ----------------- PDF analysis -----------------

def analyze_pdf(filepath):
    """Extract text and metadata from PDF."""
    try:
        with pdfplumber.open(filepath) as pdf:
            num_pages = len(pdf.pages)
            text_parts = []
            total_chars = 0
            total_words = 0
            for i, page in enumerate(pdf.pages[:50]):  # Cap to first 50 pages
                page_text = page.extract_text() or ''
                text_parts.append(page_text)
                total_chars += len(page_text)
                total_words += len(page_text.split())

            full_text = '\n\n'.join(text_parts)
            return {
                'num_pages': num_pages,
                'pages_analyzed': min(num_pages, 50),
                'total_characters': total_chars,
                'total_words': total_words,
                'preview': full_text[:3000],
                'full_text': full_text,
            }
    except Exception as e:
        return {'error': str(e)}


# ----------------- Text analysis -----------------

def analyze_text(filepath):
    """Analyze a plain text file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        lines = content.split('\n')
        words = content.split()
        return {
            'characters': len(content),
            'words': len(words),
            'lines': len(lines),
            'unique_words': len(set(w.lower() for w in words)),
            'preview': content[:3000],
            'full_text': content,
        }
    except Exception as e:
        return {'error': str(e)}
