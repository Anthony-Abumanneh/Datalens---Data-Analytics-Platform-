"""
Visualization module — generates interactive Plotly charts.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import plotly


def _to_json(fig):
    """Serialize Plotly figure to JSON."""
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _apply_layout(fig, title=None):
    """Apply consistent styling."""
    fig.update_layout(
        title=title,
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family='Inter, system-ui, sans-serif', size=12),
        height=450,
    )
    return fig


def histogram(df, column, bins=30):
    """Distribution histogram for a numeric column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    fig = px.histogram(df, x=column, nbins=bins, color_discrete_sequence=['#6366f1'])
    return _to_json(_apply_layout(fig, f"Distribution of {column}"))


def bar_chart(df, x_col, y_col=None, agg='count'):
    """Bar chart — counts of x, or aggregated y by x."""
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found")

    if y_col is None or agg == 'count':
        counts = df[x_col].value_counts().head(30).reset_index()
        counts.columns = [x_col, 'count']
        fig = px.bar(counts, x=x_col, y='count', color_discrete_sequence=['#6366f1'])
        title = f"Count by {x_col}"
    else:
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found")
        agg_func = {'sum': 'sum', 'mean': 'mean', 'median': 'median', 'max': 'max', 'min': 'min'}.get(agg, 'sum')
        grouped = df.groupby(x_col)[y_col].agg(agg_func).reset_index().head(30)
        fig = px.bar(grouped, x=x_col, y=y_col, color_discrete_sequence=['#6366f1'])
        title = f"{agg.title()} of {y_col} by {x_col}"

    return _to_json(_apply_layout(fig, title))


def line_chart(df, x_col, y_col):
    """Line chart of y vs x."""
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Column not found")
    sorted_df = df.sort_values(x_col)
    fig = px.line(sorted_df, x=x_col, y=y_col, color_discrete_sequence=['#6366f1'])
    return _to_json(_apply_layout(fig, f"{y_col} over {x_col}"))


def scatter_plot(df, x_col, y_col, color_col=None):
    """Scatter plot with optional color grouping."""
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Column not found")
    kwargs = {'x': x_col, 'y': y_col}
    if color_col and color_col in df.columns:
        kwargs['color'] = color_col
    # Sample if too many points
    plot_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
    fig = px.scatter(plot_df, **kwargs, color_discrete_sequence=px.colors.qualitative.Set2)
    return _to_json(_apply_layout(fig, f"{y_col} vs {x_col}"))


def box_plot(df, column, group_by=None):
    """Box plot showing distribution + outliers."""
    kwargs = {'y': column}
    if group_by and group_by in df.columns:
        kwargs['x'] = group_by
    fig = px.box(df, **kwargs, color_discrete_sequence=['#6366f1'])
    title = f"Box plot of {column}" + (f" by {group_by}" if group_by else "")
    return _to_json(_apply_layout(fig, title))


def correlation_heatmap(df):
    """Correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for correlation")
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect='auto',
    )
    return _to_json(_apply_layout(fig, "Correlation Matrix"))


def pie_chart(df, column):
    """Pie chart of category distribution."""
    counts = df[column].value_counts().head(10)
    fig = px.pie(values=counts.values, names=counts.index, color_discrete_sequence=px.colors.qualitative.Set2)
    return _to_json(_apply_layout(fig, f"Distribution of {column}"))


def get_chart_options(df):
    """
    Return suggested chart configurations based on the dataframe.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    return {
        'numeric_columns': numeric_cols,
        'categorical_columns': cat_cols,
        'date_columns': date_cols,
        'all_columns': df.columns.tolist(),
    }
