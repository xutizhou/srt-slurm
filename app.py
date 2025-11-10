"""
Streamlit Dashboard for Benchmark Log Analysis
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import re
import logging
from pathlib import Path
from collections import defaultdict

# Configure logging to show validation warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from utils import (
    find_all_runs,
    runs_to_dataframe,
    get_pareto_data,
    get_summary_stats,
    format_config_for_display,
    parse_command_line_from_err,
)
from utils.config_reader import (
    get_all_configs,
    get_server_config_details,
    get_environment_variables,
)
from utils.log_parser import parse_all_err_files, get_node_label
from utils.run_comparison import (
    compare_configs,
    compare_metrics,
    calculate_summary_scorecard,
    get_delta_data_for_graphs,
)


# Page configuration
st.set_page_config(
    page_title="Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(logs_dir):
    """Load and cache benchmark data."""
    return find_all_runs(logs_dir)


@st.cache_data(show_spinner=False)
def load_node_metrics(run_path: str):
    """Load and cache node metrics from .err files.

    Args:
        run_path: Path to the run directory

    Returns:
        List of parsed node metrics
    """
    return parse_all_err_files(run_path)


def group_nodes_by_dp(node_metrics_list):
    """Group nodes by DP index and average their metrics across TP workers.

    Args:
        node_metrics_list: List of node data dictionaries

    Returns:
        List of grouped node data, one entry per DP group with averaged metrics
    """
    # Group nodes by (run_id, DP index)
    dp_groups = defaultdict(list)

    for node_data in node_metrics_list:
        # Get DP indices from batch data
        if not node_data['prefill_batches']:
            continue

        # Use first batch's DP value as the group key
        first_dp = node_data['prefill_batches'][0].get('dp', 0)
        run_id = node_data.get('run_id', 'Unknown')
        group_key = (run_id, first_dp)

        dp_groups[group_key].append(node_data)

    # Create averaged node data for each DP group
    grouped_nodes = []

    for (run_id, dp_idx), nodes in dp_groups.items():
        if not nodes:
            continue

        # Collect all timestamps and metrics across nodes in this DP group
        all_batches = defaultdict(list)  # timestamp -> list of metric values

        for node in nodes:
            for batch in node['prefill_batches']:
                ts = batch.get('timestamp', '')
                if ts:
                    all_batches[ts].append(batch)

        # Average metrics at each timestamp
        averaged_batches = []
        for timestamp in sorted(all_batches.keys()):
            batches_at_time = all_batches[timestamp]

            # Average all numeric metrics
            avg_batch = {'timestamp': timestamp, 'dp': dp_idx}

            # List of metrics to average
            metrics = ['input_throughput', 'gen_throughput', 'new_seq', 'new_token',
                      'running_req', 'queue_req', 'inflight_req', 'transfer_req',
                      'prealloc_req', 'num_tokens', 'token_usage', 'preallocated_usage']

            for metric in metrics:
                values = [b.get(metric) for b in batches_at_time if metric in b]
                if values:
                    avg_batch[metric] = np.mean(values)

            # Copy type from first batch
            if batches_at_time:
                avg_batch['type'] = batches_at_time[0].get('type', 'prefill')

            averaged_batches.append(avg_batch)

        # Create grouped node data structure
        # Count TP workers in this group
        tp_count = len(nodes)
        worker_type = nodes[0]['node_info']['worker_type']

        grouped_node = {
            'node_info': {
                'node': f"{run_id}_DP{dp_idx}",
                'worker_type': worker_type,
                'worker_id': f"avg_{tp_count}_workers"
            },
            'prefill_batches': averaged_batches,
            'memory_snapshots': [],  # Not averaged for now
            'config': nodes[0]['config'],
            'run_id': run_id
        }

        grouped_nodes.append(grouped_node)

    return grouped_nodes


@st.cache_data(show_spinner=False)
def create_node_throughput_graph(_node_metrics_list, group_by_dp=False):
    """Create input throughput over time graph for all nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract data
        timestamps = []
        throughputs = []

        for batch in node_data['prefill_batches']:
            if 'timestamp' in batch and 'input_throughput' in batch:
                timestamps.append(batch['timestamp'])
                throughputs.append(batch['input_throughput'])

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=throughputs,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>" +
                              "Sample: %{x}<br>" +
                              "Throughput: %{y:.2f} token/s<extra></extra>"
            ))

    fig.update_layout(
        title="Input Throughput Over Time by Node",
        xaxis_title="Sample Number",
        yaxis_title="Input Throughput (tokens/s)",
        hovermode='closest',
        height=500,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


@st.cache_data(show_spinner=False)
def create_cache_hit_rate_graph(_node_metrics_list, group_by_dp=False):
    """Create cache hit rate visualization for prefill nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Calculate cache hit rate
        timestamps = []
        hit_rates = []

        for batch in node_data['prefill_batches']:
            if batch.get('type') == 'prefill' and 'new_token' in batch and 'cached_token' in batch:
                new_tokens = batch['new_token']
                cached_tokens = batch['cached_token']
                total_tokens = new_tokens + cached_tokens

                if total_tokens > 0:
                    hit_rate = (cached_tokens / total_tokens) * 100
                    timestamps.append(batch['timestamp'])
                    hit_rates.append(hit_rate)

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=hit_rates,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>" +
                              "Sample: %{x}<br>" +
                              "Cache Hit Rate: %{y:.1f}%<extra></extra>"
            ))

    fig.update_layout(
        title="Cache Hit Rate Over Time",
        xaxis_title="Sample Number",
        yaxis_title="Cache Hit Rate (%)",
        hovermode='closest',
        height=400,
        template="plotly_white",
        yaxis=dict(range=[0, 100])  # Percentage range
    )

    return fig


@st.cache_data(show_spinner=False)
def create_kv_cache_utilization_graph(_node_metrics_list, group_by_dp=False):
    """Create KV cache utilization visualization for nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract token usage (already in percentage form)
        timestamps = []
        utilization = []

        for batch in node_data['prefill_batches']:
            if 'token_usage' in batch:
                timestamps.append(batch['timestamp'])
                utilization.append(batch['token_usage'] * 100)  # Convert to percentage

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=utilization,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>" +
                              "Sample: %{x}<br>" +
                              "KV Cache Utilization: %{y:.2f}%<extra></extra>"
            ))

    fig.update_layout(
        title="KV Cache Utilization Over Time",
        xaxis_title="Sample Number",
        yaxis_title="Utilization (%)",
        hovermode='closest',
        height=400,
        template="plotly_white",
        yaxis=dict(range=[0, 100])  # Percentage range
    )

    return fig


@st.cache_data(show_spinner=False)
def create_queue_depth_graph(_node_metrics_list, group_by_dp=False):
    """Create queue depth visualization for nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract queue depth
        timestamps = []
        queue_depths = []

        for batch in node_data['prefill_batches']:
            if 'queue_req' in batch:
                timestamps.append(batch['timestamp'])
                queue_depths.append(batch['queue_req'])

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=queue_depths,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>" +
                              "Sample: %{x}<br>" +
                              "Queue Depth: %{y}<extra></extra>"
            ))

    fig.update_layout(
        title="Queue Depth Over Time",
        xaxis_title="Sample Number",
        yaxis_title="Number of Requests",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )

    return fig


@st.cache_data(show_spinner=False)
def create_node_inflight_requests_graph(_node_metrics_list, group_by_dp=False):
    """Create inflight requests visualization for prefill nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract inflight metrics
        timestamps = []
        inflight = []

        for batch in node_data['prefill_batches']:
            if 'timestamp' in batch and 'inflight_req' in batch:
                timestamps.append(batch['timestamp'])
                inflight.append(batch['inflight_req'])

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=inflight,
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                stackgroup='one',
            ))

    fig.update_layout(
        title="Inflight Requests Over Time",
        xaxis_title="Sample Number",
        yaxis_title="Number of Requests",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )

    return fig


@st.cache_data(show_spinner=False)
def create_decode_running_requests_graph(_node_metrics_list, group_by_dp=False):
    """Create running requests visualization for decode nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract running requests from decode batches only
        timestamps = []
        running_reqs = []

        for batch in node_data['prefill_batches']:
            if batch.get('type') == 'decode' and 'running_req' in batch:
                timestamps.append(batch['timestamp'])
                running_reqs.append(batch['running_req'])

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=running_reqs,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>" +
                              "Sample: %{x}<br>" +
                              "Running Requests: %{y}<extra></extra>",
                stackgroup=None
            ))

    fig.update_layout(
        title="Running Requests Over Time",
        xaxis_title="Sample Number",
        yaxis_title="Number of Requests",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )

    return fig


@st.cache_data(show_spinner=False)
def create_decode_gen_throughput_graph(_node_metrics_list, group_by_dp=False):
    """Create generation throughput visualization for decode nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract gen throughput from decode batches only
        timestamps = []
        throughput = []

        for batch in node_data['prefill_batches']:
            if batch.get('type') == 'decode' and 'gen_throughput' in batch:
                timestamps.append(batch['timestamp'])
                throughput.append(batch['gen_throughput'])

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=throughput,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{label}</b><br>" +
                              "Sample: %{x}<br>" +
                              "Gen Throughput: %{y:.2f} token/s<extra></extra>"
            ))

    fig.update_layout(
        title="Generation Throughput Over Time",
        xaxis_title="Sample Number",
        yaxis_title="Gen Throughput (tokens/s)",
        hovermode='closest',
        height=400,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


@st.cache_data(show_spinner=False)
def create_decode_transfer_req_graph(_node_metrics_list, group_by_dp=False):
    """Create transfer requests visualization for decode nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract transfer req from decode batches only
        timestamps = []
        transfer = []

        for batch in node_data['prefill_batches']:
            if batch.get('type') == 'decode' and 'transfer_req' in batch:
                timestamps.append(batch['timestamp'])
                transfer.append(batch['transfer_req'])

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=transfer,
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                stackgroup='one',
            ))

    fig.update_layout(
        title="Transfer Requests Over Time",
        xaxis_title="Sample Number",
        yaxis_title="Number of Requests",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )

    return fig


@st.cache_data(show_spinner=False)
def create_decode_disagg_stacked_graph(_node_metrics_list, group_by_dp=False):
    """Create stacked area chart for disaggregation request flow in decode nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()

    # For stacked charts, we'll show aggregated data across all nodes
    # Collect all data points by timestamp
    from collections import defaultdict
    data_by_time = defaultdict(lambda: {'running': 0, 'transfer': 0, 'prealloc': 0})

    for node_data in node_metrics_list:
        if not node_data['prefill_batches']:
            continue

        for batch in node_data['prefill_batches']:
            if batch.get('type') == 'decode':
                ts = batch.get('timestamp', '')
                if ts:
                    if 'running_req' in batch:
                        data_by_time[ts]['running'] += batch['running_req']
                    if 'transfer_req' in batch:
                        data_by_time[ts]['transfer'] += batch['transfer_req']
                    if 'prealloc_req' in batch:
                        data_by_time[ts]['prealloc'] += batch['prealloc_req']

    # Sort by timestamp and create arrays
    sorted_times = sorted(data_by_time.keys())
    x_vals = list(range(len(sorted_times)))

    running_vals = [data_by_time[ts]['running'] for ts in sorted_times]
    transfer_vals = [data_by_time[ts]['transfer'] for ts in sorted_times]
    prealloc_vals = [data_by_time[ts]['prealloc'] for ts in sorted_times]

    # Add traces in reverse order (bottom to top of stack)
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=prealloc_vals,
        mode='lines',
        name='Prealloc Queue',
        line=dict(width=0),
        fillcolor='rgba(99, 110, 250, 0.3)',
        stackgroup='one',
        hovertemplate='Prealloc: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=transfer_vals,
        mode='lines',
        name='Transfer Queue',
        line=dict(width=0),
        fillcolor='rgba(239, 85, 59, 0.3)',
        stackgroup='one',
        hovertemplate='Transfer: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=running_vals,
        mode='lines',
        name='Running',
        line=dict(width=0),
        fillcolor='rgba(0, 204, 150, 0.3)',
        stackgroup='one',
        hovertemplate='Running: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title="Disaggregation Request Flow (Stacked)",
        xaxis_title="Sample Number",
        yaxis_title="Number of Requests",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )

    return fig


@st.cache_data(show_spinner=False)
def create_decode_prealloc_req_graph(_node_metrics_list, group_by_dp=False):
    """Create prealloc requests visualization for decode nodes.

    Args:
        _node_metrics_list: List of node data (prefixed with _ to skip hashing)
        group_by_dp: If True, group nodes by DP and show averaged lines
    """
    # Group nodes if requested
    if group_by_dp:
        node_metrics_list = group_nodes_by_dp(_node_metrics_list)
    else:
        node_metrics_list = _node_metrics_list

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, node_data in enumerate(node_metrics_list):
        if not node_data['prefill_batches']:
            continue

        label = get_node_label(node_data)
        color = colors[idx % len(colors)]

        # Extract prealloc req from decode batches only
        timestamps = []
        prealloc = []

        for batch in node_data['prefill_batches']:
            if batch.get('type') == 'decode' and 'prealloc_req' in batch:
                timestamps.append(batch['timestamp'])
                prealloc.append(batch['prealloc_req'])

        if timestamps:
            fig.add_trace(go.Scatter(
                x=list(range(len(timestamps))),
                y=prealloc,
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                stackgroup='one',
            ))

    fig.update_layout(
        title="Prealloc Requests Over Time",
        xaxis_title="Sample Number",
        yaxis_title="Number of Requests",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )

    return fig






@st.cache_data(show_spinner=False)
def create_latency_vs_concurrency_graph(df, selected_runs, metric_name, metric_col, y_label):
    """Create latency vs concurrency graph for a specific metric.

    Args:
        df: DataFrame with benchmark data
        selected_runs: List of run IDs to plot
        metric_name: Display name for the metric (e.g., "TTFT")
        metric_col: Column name in dataframe (e.g., "Mean TTFT (ms)")
        y_label: Y-axis label
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for idx, run_id in enumerate(selected_runs):
        run_data = df[df["Run ID"] == run_id].sort_values("Concurrency")

        if len(run_data) == 0:
            continue

        color = colors[idx % len(colors)]

        # Filter out N/A values
        valid_data = run_data[run_data[metric_col] != "N/A"].copy()

        if len(valid_data) == 0:
            continue

        # Get run date if available
        run_date = valid_data.iloc[0].get('Run Date', 'N/A') if len(valid_data) > 0 else 'N/A'

        # Create hover text
        hover_text = [
            f"<b>{run_id}</b><br>" +
            f"Date: {run_date}<br>" +
            f"Concurrency: {row['Concurrency']}<br>" +
            f"{metric_name}: {row[metric_col]:.2f} ms"
            for _, row in valid_data.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=valid_data["Concurrency"],
            y=valid_data[metric_col],
            mode='lines+markers',
            name=run_id,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
            hovertext=hover_text,
            hoverinfo='text'
        ))

    fig.update_layout(
        title={
            'text': f"{metric_name} vs Concurrency",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Concurrency",
        yaxis_title=y_label,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=450,
        template="plotly_white"
    )

    return fig


def calculate_pareto_frontier(df):
    """Calculate the Pareto frontier points.

    A point is on the Pareto frontier if no other point is strictly better
    in both dimensions (higher TPS/User AND higher TPS/GPU).

    Args:
        df: DataFrame with 'Output TPS/User' and 'Output TPS/GPU' columns

    Returns:
        List of (x, y) tuples representing frontier points, sorted by x
    """
    points = df[['Output TPS/User', 'Output TPS/GPU']].values.tolist()

    if len(points) == 0:
        return []

    # Find all non-dominated points
    frontier = []

    for i, (x1, y1) in enumerate(points):
        is_dominated = False

        # Check if this point is dominated by any other point
        for j, (x2, y2) in enumerate(points):
            if i != j:
                # Point (x2, y2) dominates (x1, y1) if it's better in both dimensions
                if x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                    is_dominated = True
                    break

        if not is_dominated:
            frontier.append((x1, y1))

    # Sort frontier points by x coordinate for proper line drawing
    frontier_sorted = sorted(frontier, key=lambda p: p[0])

    return frontier_sorted


@st.cache_data(show_spinner=False)
def create_pareto_graph(df, selected_runs, show_cutoff=False, cutoff_value=30.0, show_frontier=False):
    """Create interactive Pareto graph with optional cutoff line and frontier.

    Args:
        df: DataFrame with benchmark data
        selected_runs: List of run IDs to plot
        show_cutoff: Whether to show vertical cutoff line
        cutoff_value: X-axis value for cutoff line (TPS/User)
        show_frontier: Whether to show the Pareto frontier
    """
    fig = go.Figure()

    # Add Pareto frontier FIRST if enabled (so it appears behind data points)
    if show_frontier and len(df) > 0:
        frontier_points = calculate_pareto_frontier(df)

        if len(frontier_points) > 1:  # Need at least 2 points to draw a line
            frontier_x = [p[0] for p in frontier_points]
            frontier_y = [p[1] for p in frontier_points]

            # Outer glow - widest, most transparent
            fig.add_trace(go.Scatter(
                x=frontier_x,
                y=frontier_y,
                mode='lines',
                line=dict(
                    color='rgba(255, 223, 0, 0.15)',  # Very light gold
                    width=20
                ),
                showlegend=False,
                hoverinfo='skip',
                name='Frontier Glow Outer'
            ))

            # Middle glow
            fig.add_trace(go.Scatter(
                x=frontier_x,
                y=frontier_y,
                mode='lines',
                line=dict(
                    color='rgba(255, 223, 0, 0.25)',  # Light gold
                    width=12
                ),
                showlegend=False,
                hoverinfo='skip',
                name='Frontier Glow Middle'
            ))

            # Thin highlighted line
            fig.add_trace(go.Scatter(
                x=frontier_x,
                y=frontier_y,
                mode='lines',
                line=dict(
                    color='rgba(255, 215, 0, 0.6)',  # Semi-transparent gold
                    width=3,
                    dash='dot'
                ),
                showlegend=False,
                hoverinfo='skip',
                name='Frontier Line'
            ))

            # Small markers on frontier points
            fig.add_trace(go.Scatter(
                x=frontier_x,
                y=frontier_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(255, 215, 0, 0.4)',
                    line=dict(width=1, color='rgba(255, 215, 0, 0.8)')
                ),
                showlegend=False,
                hoverinfo='skip',
                name='Frontier Points'
            ))

            # Add frontier label
            mid_idx = len(frontier_points) // 2
            fig.add_annotation(
                x=frontier_x[mid_idx],
                y=frontier_y[mid_idx],
                text="⭐ Pareto Frontier",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(255, 215, 0, 0.8)",
                ax=50,
                ay=-50,
                font=dict(size=12, color="gold"),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor="rgba(255, 215, 0, 0.8)",
                borderwidth=2,
                borderpad=4
            )
        elif len(frontier_points) == 1:
            # Single point frontier - just highlight it with a subtle ring
            fig.add_trace(go.Scatter(
                x=[frontier_points[0][0]],
                y=[frontier_points[0][1]],
                mode='markers',
                marker=dict(
                    size=25,
                    color='rgba(255, 215, 0, 0.3)',
                    line=dict(width=2, color='rgba(255, 215, 0, 0.6)')
                ),
                showlegend=False,
                hoverinfo='skip',
                name='Frontier Point'
            ))

    # Now add the actual data points (so they appear on top of frontier)
    colors = px.colors.qualitative.Set1
    for idx, run_id in enumerate(selected_runs):
        run_data = df[df["Run ID"] == run_id]

        fig.add_trace(go.Scatter(
            x=run_data["Output TPS/User"],
            y=run_data["Output TPS/GPU"],
            mode='markers+lines',
            name=f"Run {run_id}",
            marker=dict(size=10, color=colors[idx % len(colors)]),
            line=dict(color=colors[idx % len(colors)], width=2),
            text=[
                f"Run: {row['Run ID']}<br>"
                f"Concurrency: {row['Concurrency']}<br>"
                f"Output TPS/User: {row['Output TPS/User']:.2f}<br>"
                f"Output TPS/GPU: {row['Output TPS/GPU']:.2f}<br>"
                f"Output TPS: {row['Output TPS']:.2f}<br>"
                f"Mean TTFT: {row['Mean TTFT (ms)']:.2f} ms<br>"
                f"Mean TPOT: {row['Mean TPOT (ms)']:.2f} ms"
                for _, row in run_data.iterrows()
            ],
            hoverinfo='text'
        ))

    # Add vertical cutoff line if enabled
    if show_cutoff:
        fig.add_vline(
            x=cutoff_value,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Target: {cutoff_value} TPS/User",
            annotation_position="top right",
            annotation_font_size=12,
            annotation_font_color="red"
        )

    fig.update_layout(
        title={
            'text': "Pareto Frontier: Output TPS/GPU vs Output TPS/User",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Output TPS/User",
        yaxis_title="Output TPS/GPU",
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=600,
        template="plotly_white"
    )

    return fig




def main():
    # Header
    st.markdown('<div class="main-header">📊 Benchmark Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Interactive visualization and analysis of benchmark logs")

    # Sidebar - Directory selection and filters
    st.sidebar.header("Configuration")

    # Directory input
    default_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = st.sidebar.text_input(
        "Logs Directory Path",
        value=default_dir,
        help="Path to the directory containing benchmark log folders"
    )

    if not os.path.exists(logs_dir):
        st.error(f"Directory not found: {logs_dir}")
        return

    # Load data
    with st.spinner("Loading benchmark data..."):
        runs = load_data(logs_dir)

    if not runs:
        st.warning("No benchmark runs found in the specified directory.")
        st.info("Make sure the directory contains subdirectories with benchmark results.")
        return

    # Run selection with formatted labels
    st.sidebar.header("Run Selection")

    # Always sort by date, newest first
    sorted_runs = sorted(runs.copy(), key=lambda r: r.get('run_date', ''), reverse=True)

    # Add filtering options
    st.sidebar.subheader("Filters")

    # 1. Date Range Filter
    with st.sidebar.expander("📅 Date Range", expanded=False):
        # Get min/max dates from runs
        dates_with_data = [r.get('run_date') for r in sorted_runs if r.get('run_date') and r.get('run_date') != 'N/A']

        if dates_with_data:
            from datetime import datetime, timedelta

            date_objects = []
            for date_str in dates_with_data:
                try:
                    date_objects.append(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S"))
                except:
                    pass

            if date_objects:
                min_date = min(date_objects).date()
                max_date = max(date_objects).date()

                date_filter_option = st.radio(
                    "Select date range",
                    options=["All time", "Last 7 days", "Last 30 days", "Custom range"],
                    key="date_filter_option"
                )

                if date_filter_option == "Last 7 days":
                    filter_start_date = max_date - timedelta(days=7)
                    filter_end_date = max_date
                elif date_filter_option == "Last 30 days":
                    filter_start_date = max_date - timedelta(days=30)
                    filter_end_date = max_date
                elif date_filter_option == "Custom range":
                    filter_start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="date_from")
                    filter_end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="date_to")
                else:  # All time
                    filter_start_date = None
                    filter_end_date = None

                # Apply date filter
                if filter_start_date and filter_end_date:
                    filtered_by_date = []
                    for run in sorted_runs:
                        run_date_str = run.get('run_date')
                        if run_date_str and run_date_str != 'N/A':
                            try:
                                run_date = datetime.strptime(run_date_str, "%Y-%m-%d %H:%M:%S").date()
                                if filter_start_date <= run_date <= filter_end_date:
                                    filtered_by_date.append(run)
                            except:
                                pass
                    sorted_runs = filtered_by_date
        else:
            st.caption("No date information available")

    # 2. Topology Filter
    with st.sidebar.expander("🔧 Topology", expanded=False):
        # Extract unique topologies
        topologies = set()
        for run in sorted_runs:
            prefill_dp = run.get('prefill_dp', '?')
            decode_dp = run.get('decode_dp', '?')
            topology = f"{prefill_dp}P/{decode_dp}D"
            if topology != "?P/?D":
                topologies.add(topology)

        if topologies:
            topology_options = sorted(list(topologies))
            selected_topologies = st.multiselect(
                "Select topologies",
                options=topology_options,
                default=topology_options,
                key="topology_filter"
            )

            # Apply topology filter
            if selected_topologies:
                filtered_by_topology = []
                for run in sorted_runs:
                    topology = f"{run.get('prefill_dp', '?')}P/{run.get('decode_dp', '?')}D"
                    if topology in selected_topologies:
                        filtered_by_topology.append(run)
                sorted_runs = filtered_by_topology
        else:
            st.caption("No topology information available")

    # 3. ISL/OSL Filter
    with st.sidebar.expander("📊 ISL/OSL", expanded=False):
        # Extract unique ISL and OSL values
        isl_values = set()
        osl_values = set()
        for run in sorted_runs:
            isl = run.get('isl')
            osl = run.get('osl')
            if isl and isl != 'N/A' and isl != '?':
                isl_values.add(isl)
            if osl and osl != 'N/A' and osl != '?':
                osl_values.add(osl)

        if isl_values:
            isl_options = sorted(list(isl_values))
            selected_isl = st.multiselect(
                "Input Sequence Length (ISL)",
                options=isl_options,
                default=isl_options,
                key="isl_filter"
            )
        else:
            selected_isl = None
            st.caption("No ISL information available")

        if osl_values:
            osl_options = sorted(list(osl_values))
            selected_osl = st.multiselect(
                "Output Sequence Length (OSL)",
                options=osl_options,
                default=osl_options,
                key="osl_filter"
            )
        else:
            selected_osl = None
            st.caption("No OSL information available")

        # Apply ISL/OSL filter
        if selected_isl or selected_osl:
            filtered_by_isl_osl = []
            for run in sorted_runs:
                isl_match = (not selected_isl) or (run.get('isl') in selected_isl)
                osl_match = (not selected_osl) or (run.get('osl') in selected_osl)
                if isl_match and osl_match:
                    filtered_by_isl_osl.append(run)
            sorted_runs = filtered_by_isl_osl

    # 4. Container Filter
    with st.sidebar.expander("🐳 Container", expanded=False):
        # Extract unique containers
        container_values = set()
        for run in sorted_runs:
            container = run.get('container')
            if container and container != 'N/A':
                container_values.add(container)

        if container_values:
            container_options = sorted(list(container_values))
            selected_containers = st.multiselect(
                "Select containers",
                options=container_options,
                default=container_options,
                key="container_filter"
            )

            # Apply container filter
            if selected_containers:
                filtered_by_container = []
                for run in sorted_runs:
                    container = run.get('container')
                    if container in selected_containers or (not container and 'N/A' in selected_containers):
                        filtered_by_container.append(run)
                sorted_runs = filtered_by_container
        else:
            st.caption("No container information available")

    # Show filter results
    st.sidebar.caption(f"✅ {len(sorted_runs)} runs match filters")

    # Create formatted labels and mapping
    run_labels = []
    label_to_run = {}

    for run in sorted_runs:
        job_id = run.get("slurm_job_id", "Unknown")
        date = run.get("run_date", "N/A")

        # Parse date for shorter display
        if date != "N/A":
            try:
                from datetime import datetime
                date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                date_short = date_obj.strftime("%b %d").replace(' 0', ' ')  # "Nov 4"
            except:
                date_short = date.split()[0]  # Fallback to YYYY-MM-DD
        else:
            date_short = "No date"

        # Extract job number
        job_num = job_id.split('_')[0] if '_' in job_id else job_id

        # Create readable label
        topology = f"{run.get('prefill_dp', '?')}P/{run.get('decode_dp', '?')}D"
        isl = run.get('isl', '?')
        label = f"[{date_short}] Job {job_num} - {topology} (ISL {isl})"

        run_labels.append(label)
        label_to_run[label] = run

    # Multiselect with formatted labels
    selected_labels = st.sidebar.multiselect(
        "Select runs to compare",
        options=run_labels,
        default=run_labels[:min(3, len(run_labels))],  # Select first 3 by default
        help="Select one or more runs to visualize"
    )

    if not selected_labels:
        st.warning("Please select at least one run to visualize.")
        return

    # Filter runs based on selected labels
    filtered_runs = [label_to_run[label] for label in selected_labels]

    # Extract run IDs for compatibility with existing graph functions
    selected_runs = [run.get("slurm_job_id", "Unknown") for run in filtered_runs]

    # Get dataframe
    df = runs_to_dataframe(filtered_runs)

    # Filters
    st.sidebar.header("Filters")

    # Concurrency range filter
    min_concurrency = int(df["Concurrency"].min())
    max_concurrency = int(df["Concurrency"].max())

    if min_concurrency < max_concurrency:
        concurrency_range = st.sidebar.slider(
            "Concurrency Range",
            min_value=min_concurrency,
            max_value=max_concurrency,
            value=(min_concurrency, max_concurrency)
        )
        df = df[(df["Concurrency"] >= concurrency_range[0]) &
                (df["Concurrency"] <= concurrency_range[1])]

    # Pareto options
    st.sidebar.header("Pareto Graph Options")
    show_cutoff = st.sidebar.checkbox("Show TPS/User cutoff line", value=False)
    cutoff_value = st.sidebar.number_input(
        "Cutoff value (TPS/User)",
        min_value=0.0,
        max_value=1000.0,
        value=30.0,
        step=1.0,
        disabled=not show_cutoff,
        help="Vertical line to mark target TPS/User threshold"
    )
    show_frontier = st.sidebar.checkbox(
        "Show Pareto Frontier",
        value=False,
        help="Highlight the efficient frontier - points where no other configuration is strictly better"
    )

    # Summary metrics
    st.header("Summary")

    # Show unique containers
    containers = [run.get('container') for run in filtered_runs if run.get('container')]
    if containers:
        unique_containers = list(set(containers))
        if len(unique_containers) == 1:
            st.caption(f"🐳 Container: {unique_containers[0]}")
        else:
            st.caption(f"🐳 Containers: {', '.join(unique_containers)}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Runs", len(selected_runs))
    with col2:
        st.metric("Data Points", len(df))
    with col3:
        st.metric("Max Throughput", f"{df['Output TPS'].max():.0f} TPS")
    with col4:
        st.metric("Max Concurrency", int(df["Concurrency"].max()))
    with col5:
        unique_profilers = df["Profiler"].nunique()
        st.metric("Profilers", unique_profilers)

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Pareto Graph",
        "⏱️ Latency Analysis",
        "🖥️ Node Metrics",
        "⚙️ Configuration",
        "🔬 Run Comparison"
    ])

    with tab1:
        st.subheader("Pareto Frontier Analysis")
        st.markdown("""
        This graph shows the trade-off between **Output TPS/GPU** (efficiency) and
        **Output TPS/User** (throughput per user).
        """)

        pareto_fig = create_pareto_graph(df, selected_runs, show_cutoff, cutoff_value, show_frontier)
        pareto_fig.update_xaxes(showgrid=True)
        pareto_fig.update_yaxes(showgrid=True)

        st.plotly_chart(pareto_fig, width="stretch", key="pareto_main")

        # Debug info for frontier
        if show_frontier:
            frontier_points = calculate_pareto_frontier(df)
            st.caption(f"🔍 Debug: Frontier has {len(frontier_points)} points across {len(df)} total data points")

            # Show which points are on the frontier
            if len(frontier_points) > 0:
                with st.expander("View Frontier Points Details"):
                    frontier_df = pd.DataFrame(frontier_points, columns=['Output TPS/User', 'Output TPS/GPU'])
                    st.dataframe(frontier_df, width="stretch")

        # Add data export button below the graph
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            st.download_button(
                label="📥 Download Data as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="benchmark_data.csv",
                mime="text/csv",
                width="stretch"
            )

        # Metric calculation documentation
        st.divider()
        st.markdown("### 📊 Metric Calculations")
        st.markdown("""
        **How each metric is calculated:**

        **Output TPS/GPU** (Throughput Efficiency):
        """)
        st.latex(r"\text{Output TPS/GPU} = \frac{\text{Total Output Throughput (tokens/s)}}{\text{Total Number of GPUs}}")
        st.markdown("""
        *This measures how efficiently each GPU is being utilized for token generation.*

        **Output TPS/User** (Per-User Generation Rate):
        """)
        st.latex(r"\text{Output TPS/User} = \frac{1000}{\text{Mean TPOT (ms)}}")
        st.markdown("""
        *Where TPOT (Time Per Output Token) is the average time between consecutive output tokens.
        This represents the actual token generation rate experienced by each user, independent of concurrency.*
        """)

    with tab2:
        st.subheader("Latency Analysis")

        if len(selected_runs) == 0:
            st.warning("Please select at least one run from the sidebar.")
        else:
            # TTFT Graph
            st.markdown("### Time to First Token (TTFT)")
            st.markdown("""
            **TTFT** measures the time from request submission to receiving the first output token.
            This is critical for perceived responsiveness.
            """)
            ttft_fig = create_latency_vs_concurrency_graph(
                df, selected_runs,
                metric_name="TTFT",
                metric_col="Mean TTFT (ms)",
                y_label="Mean TTFT (ms)"
            )
            st.plotly_chart(ttft_fig, width="stretch", key="latency_ttft")

            # TPOT Graph
            st.markdown("### Time Per Output Token (TPOT)")
            st.markdown("""
            **TPOT** measures the time between consecutive output tokens during generation.
            Lower TPOT means faster streaming and better user experience.
            """)
            tpot_fig = create_latency_vs_concurrency_graph(
                df, selected_runs,
                metric_name="TPOT",
                metric_col="Mean TPOT (ms)",
                y_label="Mean TPOT (ms)"
            )
            st.plotly_chart(tpot_fig, width="stretch", key="latency_tpot")

            # ITL Graph
            st.markdown("### Inter-Token Latency (ITL)")
            st.markdown("""
            **ITL** measures the interval between tokens during generation.
            Similar to TPOT but may include queueing delays.
            """)
            itl_fig = create_latency_vs_concurrency_graph(
                df, selected_runs,
                metric_name="ITL",
                metric_col="Mean ITL (ms)",
                y_label="Mean ITL (ms)"
            )
            st.plotly_chart(itl_fig, width="stretch", key="latency_itl")

            # Summary statistics
            st.divider()
            st.markdown("### 📊 Latency Summary Statistics")

            summary_data = []
            for run_id in selected_runs:
                run_data = df[df["Run ID"] == run_id]
                if len(run_data) > 0:
                    summary_data.append({
                        "Run ID": run_id,
                        "Min TTFT (ms)": run_data["Mean TTFT (ms)"].min() if "Mean TTFT (ms)" in run_data.columns else "N/A",
                        "Max TTFT (ms)": run_data["Mean TTFT (ms)"].max() if "Mean TTFT (ms)" in run_data.columns else "N/A",
                        "Min TPOT (ms)": run_data["Mean TPOT (ms)"].min() if "Mean TPOT (ms)" in run_data.columns else "N/A",
                        "Max TPOT (ms)": run_data["Mean TPOT (ms)"].max() if "Mean TPOT (ms)" in run_data.columns else "N/A",
                        "Min ITL (ms)": run_data["Mean ITL (ms)"].min() if "Mean ITL (ms)" in run_data.columns else "N/A",
                        "Max ITL (ms)": run_data["Mean ITL (ms)"].max() if "Mean ITL (ms)" in run_data.columns else "N/A",
                    })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width="stretch")

    with tab3:
        st.subheader("Node-Level Metrics")
        st.markdown("""
        Runtime metrics extracted from log files, split by prefill and decode nodes.
        """)

        # Parse log files for all selected runs (cached)
        all_node_metrics = []
        for run in filtered_runs:
            run_path = run.get("path", "")
            run_id = run.get("slurm_job_id", "Unknown")
            if run_path and os.path.exists(run_path):
                node_metrics = load_node_metrics(run_path)
                # Add run_id to each node for identification in multi-run comparisons
                for node_data in node_metrics:
                    node_data['run_id'] = run_id
                all_node_metrics.extend(node_metrics)

        if not all_node_metrics:
            st.warning("No log files (.err) found for the selected runs.")
            st.info("Node metrics are extracted from files like `*_prefill_*.err` and `*_decode_*.err`")
        else:
            # Split by prefill vs decode
            prefill_nodes = [n for n in all_node_metrics if n['node_info']['worker_type'] == 'prefill']
            decode_nodes = [n for n in all_node_metrics if n['node_info']['worker_type'] == 'decode']

            st.caption(f"📊 Found {len(prefill_nodes)} prefill nodes, {len(decode_nodes)} decode nodes")

            # Add toggle for grouping
            group_by_dp = st.checkbox(
                "📊 Group by DP (show averaged lines per DP group)",
                value=True,
                help="When enabled, nodes are grouped by DP index and metrics are averaged across TP workers. This reduces visual clutter when viewing many nodes."
            )

            # Prefill Metrics Section
            if prefill_nodes:
                st.markdown("### 📤 Prefill Node Metrics")

                # Vertically stack graphs for better horizontal stretching
                throughput_fig = create_node_throughput_graph(prefill_nodes, group_by_dp=group_by_dp)
                throughput_fig.update_xaxes(showgrid=True)
                throughput_fig.update_yaxes(showgrid=True)
                st.plotly_chart(throughput_fig, width="stretch", key="prefill_throughput")
                st.caption("Shows prefill throughput in tokens/s - measures how fast the system processes input prompts")

                inflight_fig = create_node_inflight_requests_graph(prefill_nodes, group_by_dp=group_by_dp)
                inflight_fig.update_xaxes(showgrid=True)
                inflight_fig.update_yaxes(showgrid=True)
                st.plotly_chart(inflight_fig, width="stretch", key="prefill_inflight")
                st.caption("Requests that have been sent to decode workers in PD disaggregation mode")

                cache_fig = create_cache_hit_rate_graph(prefill_nodes, group_by_dp=group_by_dp)
                cache_fig.update_xaxes(showgrid=True)
                cache_fig.update_yaxes(showgrid=True)
                st.plotly_chart(cache_fig, width="stretch", key="prefill_cache_hit")
                st.caption("Percentage of tokens found in prefix cache - higher values indicate better cache reuse and reduced compute")

                kv_fig = create_kv_cache_utilization_graph(prefill_nodes, group_by_dp=group_by_dp)
                kv_fig.update_xaxes(showgrid=True)
                kv_fig.update_yaxes(showgrid=True)
                st.plotly_chart(kv_fig, width="stretch", key="prefill_kv_util")
                st.caption("Percentage of KV cache memory currently in use - helps tune max-total-tokens and identify memory pressure")

                queue_fig = create_queue_depth_graph(prefill_nodes, group_by_dp=group_by_dp)
                queue_fig.update_xaxes(showgrid=True)
                queue_fig.update_yaxes(showgrid=True)
                st.plotly_chart(queue_fig, width="stretch", key="prefill_queue")
                st.caption("Number of requests waiting in queue - growing queue indicates system overload or backpressure")

            # Decode Metrics Section
            if decode_nodes:
                st.divider()
                st.markdown("### 📥 Decode Node Metrics")

                # Debug: Check if decode nodes have batch data
                has_data = any(node_data['prefill_batches'] for node_data in decode_nodes)
                if not has_data:
                    st.warning("⚠️ No batch metrics found for decode nodes. Decode nodes may not log batch-level metrics in the current setup.")
                else:
                    running_fig = create_decode_running_requests_graph(decode_nodes, group_by_dp=group_by_dp)
                    running_fig.update_xaxes(showgrid=True)
                    running_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(running_fig, width="stretch", key="decode_running")
                    st.caption("Number of requests currently being decoded and generating output tokens")

                    gen_fig = create_decode_gen_throughput_graph(decode_nodes, group_by_dp=group_by_dp)
                    gen_fig.update_xaxes(showgrid=True)
                    gen_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(gen_fig, width="stretch", key="decode_gen_throughput")
                    st.caption("Output token generation rate in tokens/s - measures decode performance")

                    kv_decode_fig = create_kv_cache_utilization_graph(decode_nodes, group_by_dp=group_by_dp)
                    kv_decode_fig.update_xaxes(showgrid=True)
                    kv_decode_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(kv_decode_fig, width="stretch", key="decode_kv_util")
                    st.caption("Total KV cache tokens in use across all running requests - low indicates underutilization, high indicates risk of OOM")

                    queue_decode_fig = create_queue_depth_graph(decode_nodes, group_by_dp=group_by_dp)
                    queue_decode_fig.update_xaxes(showgrid=True)
                    queue_decode_fig.update_yaxes(showgrid=True)
                    st.plotly_chart(queue_decode_fig, width="stretch", key="decode_queue")
                    st.caption("Number of requests waiting in decode queue - indicates decode capacity constraints")

                    # Disaggregation metrics with toggle
                    st.divider()
                    st.markdown("#### Disaggregation Metrics")

                    disagg_view = st.radio(
                        "View mode",
                        options=["Stacked (Combined)", "Separate Graphs"],
                        index=0,
                        horizontal=True,
                        help="Stacked view shows request flow through stages. Separate graphs show individual metrics."
                    )

                    if disagg_view == "Stacked (Combined)":
                        stacked_fig = create_decode_disagg_stacked_graph(decode_nodes, group_by_dp=group_by_dp)
                        stacked_fig.update_xaxes(showgrid=True)
                        stacked_fig.update_yaxes(showgrid=True)
                        st.plotly_chart(stacked_fig, width="stretch", key="decode_disagg_stacked")
                        st.caption("Shows the request flow funnel: Prealloc Queue → Transfer Queue → Running requests in PD disaggregation")
                    else:
                        transfer_fig = create_decode_transfer_req_graph(decode_nodes, group_by_dp=group_by_dp)
                        transfer_fig.update_xaxes(showgrid=True)
                        transfer_fig.update_yaxes(showgrid=True)
                        st.plotly_chart(transfer_fig, width="stretch", key="decode_transfer")
                        st.caption("Requests waiting for KV cache transfer from prefill to decode workers in PD disaggregation mode")

                        prealloc_fig = create_decode_prealloc_req_graph(decode_nodes, group_by_dp=group_by_dp)
                        prealloc_fig.update_xaxes(showgrid=True)
                        prealloc_fig.update_yaxes(showgrid=True)
                        st.plotly_chart(prealloc_fig, width="stretch", key="decode_prealloc")
                        st.caption("Requests in pre-allocation queue for PD disaggregation - waiting for memory allocation on decode workers")

    with tab4:
        st.subheader("Run Configuration Details")

        for run in filtered_runs:
            run_id = run.get("slurm_job_id", "Unknown")
            run_path = run.get("path", "")
            run_date = run.get("run_date", None)

            # Add date to expander title if available
            expander_title = f"🔧 {run_id}"
            if run_date and run_date != "N/A":
                expander_title += f" ({run_date})"

            with st.expander(expander_title, expanded=len(filtered_runs) == 1):
                if not run_path or not os.path.exists(run_path):
                    st.warning("Configuration files not found for this run.")
                    continue

                # Get structured config data
                config_data = format_config_for_display(run_path)

                if "error" in config_data:
                    st.error(config_data["error"])
                    continue

                # High-level Summary
                st.markdown("### 📋 Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Nodes", config_data["summary"]["num_nodes"])
                with col2:
                    st.metric("GPU Type", config_data["summary"]["gpu_type"])
                with col3:
                    profiler = run.get('profiler_type', 'N/A')
                    st.metric("Profiler", profiler)
                with col4:
                    st.metric("ISL / OSL", f"{run.get('isl', 'N/A')} / {run.get('osl', 'N/A')}")

                st.markdown(f"**Model:** {config_data['summary']['model']}")

                # Deployment Topology - Parse from .err files
                st.markdown("### 🚀 Deployment Topology")

                # Parse .err files to get actual services running on each node
                parsed_data = parse_command_line_from_err(run_path)
                physical_nodes = parsed_data.get('services', {})

                # Enrich with GPU info from config data
                config_node_info = {}
                all_nodes = (config_data.get("prefill_nodes", []) +
                            config_data.get("decode_nodes", []) +
                            config_data.get("frontend_nodes", []) +
                            config_data.get("other_nodes", []))

                for node in all_nodes:
                    node_name = node.get('node_name', 'Unknown')
                    config_node_info[node_name] = {
                        'gpu_count': node.get('gpu_count', 'N/A'),
                        'tp': node.get('tp_size', 'N/A'),
                        'dp': node.get('dp_size', 'N/A'),
                    }

                # Display by physical node
                if physical_nodes:
                    num_nodes = len(physical_nodes)
                    cols_per_row = 3

                    node_items = sorted(physical_nodes.items())
                    for i in range(0, num_nodes, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, (phys_node, service_types) in enumerate(node_items[i:i+cols_per_row]):
                            with cols[j]:
                                st.markdown(f"**🖥️ {phys_node}**")
                                for service_type in sorted(set(service_types)):  # Deduplicate
                                    # Add emoji based on service type
                                    emoji = {'prefill': '📤', 'decode': '📥', 'frontend': '🌐',
                                            'nginx': '🌐', 'nats': '📡', 'etcd': '🗄️'}.get(service_type, '⚙️')

                                    st.text(f"{emoji} {service_type}")

                                    # Try to find GPU info for this service
                                    # Look for matching config entry
                                    for config_name, info in config_node_info.items():
                                        if phys_node in config_name and service_type in config_name:
                                            if info['gpu_count'] != 'N/A' and info['gpu_count'] > 0:
                                                st.caption(f"  GPUs: {info['gpu_count']} | TP: {info['tp']} | DP: {info['dp']}")
                                            break
                else:
                    st.caption("No node topology information available")

                # Get detailed config from first available node
                all_configs = get_all_configs(run_path)
                if all_configs:
                    # Server Configuration - Display all flags with explicit flags first
                    st.markdown("### ⚙️ Server Configuration")
                    server_config = get_server_config_details(all_configs[0])

                    if server_config:
                        # Get explicit flags from parsed command line
                        explicit_flags = parsed_data.get('explicit_flags', set())

                        # Convert flag names: disaggregation-mode -> disaggregation_mode
                        explicit_flags_normalized = {flag.replace('-', '_') for flag in explicit_flags}

                        # Separate explicitly set flags from defaults
                        explicit_items = []
                        default_items = []

                        for key, value in sorted(server_config.items()):
                            if key in explicit_flags_normalized:
                                explicit_items.append((key, value))
                            else:
                                default_items.append((key, value))

                        # Display explicitly set flags first
                        if explicit_items:
                            st.markdown("**Explicitly Set Flags**")
                            num_items = len(explicit_items)
                            items_per_col = (num_items + 2) // 3

                            col1, col2, col3 = st.columns(3)

                            for idx, (key, value) in enumerate(explicit_items):
                                col_idx = idx // items_per_col
                                if col_idx == 0:
                                    with col1:
                                        st.caption(f"{key}: {value}")
                                elif col_idx == 1:
                                    with col2:
                                        st.caption(f"{key}: {value}")
                                else:
                                    with col3:
                                        st.caption(f"{key}: {value}")

                        # Display default flags in an expander
                        if default_items:
                            with st.expander(f"Default Flags ({len(default_items)} total)", expanded=False):
                                num_items = len(default_items)
                                items_per_col = (num_items + 2) // 3

                                col1, col2, col3 = st.columns(3)

                                for idx, (key, value) in enumerate(default_items):
                                    col_idx = idx // items_per_col
                                    if col_idx == 0:
                                        with col1:
                                            st.caption(f"{key}: {value}")
                                    elif col_idx == 1:
                                        with col2:
                                            st.caption(f"{key}: {value}")
                                    else:
                                        with col3:
                                            st.caption(f"{key}: {value}")

                    # Environment Variables per Node
                    st.markdown("### 🌍 Environment Variables")

                    # Group nodes by type for easier navigation
                    node_names = [config.get('filename', f'Node {i}').replace('_config.json', '')
                                 for i, config in enumerate(all_configs)]

                    if len(node_names) > 0:
                        prefill_configs = [(name, config) for name, config in zip(node_names, all_configs)
                                          if 'prefill' in name.lower()]
                        decode_configs = [(name, config) for name, config in zip(node_names, all_configs)
                                         if 'decode' in name.lower()]
                        other_configs = [(name, config) for name, config in zip(node_names, all_configs)
                                        if 'prefill' not in name.lower() and 'decode' not in name.lower()]

                        # Display environment variables by node type
                        if prefill_configs:
                            with st.expander("📤 Prefill Nodes", expanded=False):
                                for node_name, config in prefill_configs:
                                    st.markdown(f"**`{node_name.replace('watchtower-navy-', '')}`**")
                                    env_vars = get_environment_variables(config)
                                    if env_vars:
                                        cols = st.columns(min(len(env_vars), 3))
                                        for idx, (category, vars_dict) in enumerate(env_vars.items()):
                                            with cols[idx % len(cols)]:
                                                st.markdown(f"*{category}*")
                                                for key, value in list(vars_dict.items())[:3]:  # Show first 3
                                                    st.caption(f"{key}={value}")
                                                if len(vars_dict) > 3:
                                                    with st.expander(f"Show all {len(vars_dict)}"):
                                                        for key, value in vars_dict.items():
                                                            st.caption(f"{key}={value}")
                                    else:
                                        st.caption("No environment variables found")

                        if decode_configs:
                            with st.expander("📥 Decode Nodes", expanded=False):
                                for node_name, config in decode_configs:
                                    st.markdown(f"**`{node_name.replace('watchtower-navy-', '')}`**")
                                    env_vars = get_environment_variables(config)
                                    if env_vars:
                                        cols = st.columns(min(len(env_vars), 3))
                                        for idx, (category, vars_dict) in enumerate(env_vars.items()):
                                            with cols[idx % len(cols)]:
                                                st.markdown(f"*{category}*")
                                                for key, value in list(vars_dict.items())[:3]:
                                                    st.caption(f"{key}={value}")
                                                if len(vars_dict) > 3:
                                                    with st.expander(f"Show all {len(vars_dict)}"):
                                                        for key, value in vars_dict.items():
                                                            st.caption(f"{key}={value}")
                                    else:
                                        st.caption("No environment variables found")

                        if other_configs:
                            with st.expander("🖥️ Other Nodes", expanded=False):
                                for node_name, config in other_configs:
                                    st.markdown(f"**`{node_name.replace('watchtower-navy-', '')}`**")
                                    env_vars = get_environment_variables(config)
                                    if env_vars:
                                        for category, vars_dict in env_vars.items():
                                            st.markdown(f"*{category}*")
                                            for key, value in list(vars_dict.items())[:5]:
                                                st.caption(f"{key}={value}")
                                            if len(vars_dict) > 5:
                                                with st.expander(f"Show all {len(vars_dict)}"):
                                                    for key, value in vars_dict.items():
                                                        st.caption(f"{key}={value}")
                                    else:
                                        st.caption("No environment variables found")

    with tab5:
        st.subheader("🔬 Run Comparison (Isolation Mode)")
        st.markdown("""
        Compare two benchmark runs in detail to understand configuration changes and performance deltas.
        Select exactly 2 runs to begin comparison.
        """)

        # Run Selection Panel
        st.markdown("### 📋 Select Runs to Compare")

        # Add sort control
        sort_col1, sort_col2 = st.columns([1, 3])
        with sort_col1:
            sort_by = st.selectbox(
                "Sort by",
                options=["Newest First", "Oldest First", "Job ID"],
                key="comparison_sort"
            )

        # Sort filtered_runs based on selection
        sorted_runs = filtered_runs.copy()
        if sort_by == "Newest First":
            sorted_runs = sorted(sorted_runs, key=lambda r: r.get('run_date', ''), reverse=True)
        elif sort_by == "Oldest First":
            sorted_runs = sorted(sorted_runs, key=lambda r: r.get('run_date', ''))
        elif sort_by == "Job ID":
            sorted_runs = sorted(sorted_runs, key=lambda r: r.get('slurm_job_id', ''))

        # Create formatted labels and mapping
        run_options = []
        run_map = {}

        for run in sorted_runs:
            job_id = run.get("slurm_job_id", "Unknown")
            date = run.get("run_date", "N/A")

            # Parse date for shorter display
            if date != "N/A":
                # Format: "2025-11-04 23:18:43" -> "Nov 4"
                try:
                    from datetime import datetime
                    date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    date_short = date_obj.strftime("%b %d").replace(' 0', ' ')  # "Nov 4" (remove leading zero)
                except:
                    date_short = date.split()[0]  # Fallback to YYYY-MM-DD
            else:
                date_short = "No date"

            # Extract job number from ID like "3320_1P_4D_20251104_231843"
            job_num = job_id.split('_')[0] if '_' in job_id else job_id

            # Create readable label
            topology = f"{run.get('prefill_dp', '?')}P/{run.get('decode_dp', '?')}D"
            isl = run.get('isl', '?')
            label = f"[{date_short}] Job {job_num} - {topology} (ISL {isl})"

            run_options.append(label)
            run_map[label] = run

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Run A (Baseline)**")
            run_a_label = st.selectbox(
                "Select baseline run",
                options=run_options,
                key="run_a_select"
            )
            # Get the full run data from mapping
            run_a = run_map.get(run_a_label)

            if run_a:
                st.caption(f"📅 {run_a.get('run_date', 'N/A')}")
                st.caption(f"📊 ISL/OSL: {run_a.get('isl', 'N/A')}/{run_a.get('osl', 'N/A')}")
                st.caption(f"🖥️ Topology: {run_a.get('prefill_dp', '?')}P/{run_a.get('decode_dp', '?')}D")
                total_gpus_a = run_a.get('prefill_tp', 0) * run_a.get('prefill_dp', 0) + run_a.get('decode_tp', 0) * run_a.get('decode_dp', 0)
                st.caption(f"🎯 Total GPUs: {total_gpus_a}")

        with col2:
            st.markdown("**Run B (Comparison)**")
            run_b_label = st.selectbox(
                "Select comparison run",
                options=run_options,
                key="run_b_select"
            )
            # Get the full run data from mapping
            run_b = run_map.get(run_b_label)

            if run_b:
                st.caption(f"📅 {run_b.get('run_date', 'N/A')}")
                st.caption(f"📊 ISL/OSL: {run_b.get('isl', 'N/A')}/{run_b.get('osl', 'N/A')}")
                st.caption(f"🖥️ Topology: {run_b.get('prefill_dp', '?')}P/{run_b.get('decode_dp', '?')}D")
                total_gpus_b = run_b.get('prefill_tp', 0) * run_b.get('prefill_dp', 0) + run_b.get('decode_tp', 0) * run_b.get('decode_dp', 0)
                st.caption(f"🎯 Total GPUs: {total_gpus_b}")

        # Validation
        if run_a_label == run_b_label:
            st.warning("⚠️ Please select two different runs to compare.")
        elif run_a and run_b:
            st.divider()

            # Get config data for both runs
            run_a_path = run_a.get("path", "")
            run_b_path = run_b.get("path", "")

            configs_a = get_all_configs(run_a_path)
            configs_b = get_all_configs(run_b_path)

            if not configs_a or not configs_b:
                st.error("Could not load configuration files for one or both runs.")
            else:
                config_a = configs_a[0]
                config_b = configs_b[0]

                # Perform comparison
                config_comparison = compare_configs(config_a, config_b)
                metrics_comparison = compare_metrics(run_a, run_b)
                scorecard = calculate_summary_scorecard(metrics_comparison)

                # Display Configuration Comparison
                with st.expander("📋 Configuration Differences", expanded=True):
                    st.markdown("#### Topology Summary")

                    topo_df = pd.DataFrame({
                        'Parameter': ['Prefill TP', 'Prefill DP', 'Model', 'Context Length', 'Container'],
                        'Run A': [
                            str(config_comparison['topology_summary']['prefill_tp'][0]),
                            str(config_comparison['topology_summary']['prefill_dp'][0]),
                            str(config_comparison['topology_summary']['model'][0]),
                            str(config_comparison['topology_summary']['context_length'][0]),
                            str(run_a.get('container', 'N/A'))
                        ],
                        'Run B': [
                            str(config_comparison['topology_summary']['prefill_tp'][1]),
                            str(config_comparison['topology_summary']['prefill_dp'][1]),
                            str(config_comparison['topology_summary']['model'][1]),
                            str(config_comparison['topology_summary']['context_length'][1]),
                            str(run_b.get('container', 'N/A'))
                        ]
                    })
                    st.dataframe(topo_df, width="stretch", hide_index=True)

                    st.divider()

                    num_diffs = config_comparison['num_differences']
                    if num_diffs > 0:
                        st.markdown(f"#### ⚠️ **{num_diffs}** Configuration Flags Differ")

                        # Group by category
                        diff_df = pd.DataFrame(config_comparison['flag_differences'])

                        if not diff_df.empty:
                            for category in sorted(diff_df['category'].unique()):
                                category_diffs = diff_df[diff_df['category'] == category]
                                st.markdown(f"**{category}**")

                                display_df = category_diffs[['flag', 'run_a_value', 'run_b_value']]
                                display_df.columns = ['Flag', 'Run A', 'Run B']

                                st.dataframe(
                                    display_df,
                                    width="stretch",
                                    hide_index=True
                                )
                    else:
                        st.success("✅ No configuration differences detected!")

                    # Identical flags (collapsed)
                    num_identical = len(config_comparison['identical_flags'])
                    if num_identical > 0:
                        with st.expander(f"Show {num_identical} Identical Flags"):
                            identical_df = pd.DataFrame(config_comparison['identical_flags'])
                            # Convert all values to strings to avoid Arrow type issues
                            identical_df['value'] = identical_df['value'].astype(str)
                            st.dataframe(identical_df, width="stretch", hide_index=True)

                st.divider()

                # Display Performance Comparison
                st.markdown("### 📊 Performance Comparison")

                if not metrics_comparison.empty:
                    # Summary scorecard
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Metrics Improved", scorecard['num_improved'])
                    with col2:
                        st.metric("⚠️ Metrics Regressed", scorecard['num_regressed'])
                    with col3:
                        st.metric("➡️ Unchanged", scorecard['num_unchanged'])

                    if scorecard['biggest_improvement']:
                        st.success(f"**Biggest Win:** {scorecard['biggest_improvement']}")
                    if scorecard['biggest_regression']:
                        st.warning(f"**Biggest Loss:** {scorecard['biggest_regression']}")

                    st.divider()

                    # Detailed metrics table
                    st.markdown("#### Detailed Metrics Comparison")

                    # Format the dataframe for display
                    display_df = metrics_comparison.copy()

                    # Add improvement indicator
                    display_df['Status'] = display_df['Improved'].apply(lambda x: '✅' if x else '⚠️')

                    # Select and reorder columns
                    display_df = display_df[['Concurrency', 'Metric', 'Run A', 'Run B', 'Delta', '% Change', 'Status']]

                    # Format numeric columns
                    st.dataframe(
                        display_df.style.format({
                            'Run A': '{:.2f}',
                            'Run B': '{:.2f}',
                            'Delta': '{:.2f}',
                            '% Change': '{:.1f}%'
                        }).map(
                            lambda x: 'background-color: #d4edda' if x == '✅' else ('background-color: #f8d7da' if x == '⚠️' else ''),
                            subset=['Status']
                        ),
                        width="stretch",
                        height=400
                    )

                    # Visual Comparison Graphs
                    st.divider()
                    st.markdown("### 📈 Visual Comparison")

                    # Create sub-tabs for different visualizations
                    viz_tab1, viz_tab2 = st.tabs(["Side-by-Side Comparison", "Delta Graphs"])

                    with viz_tab1:
                        st.markdown("#### Overlay Comparison")
                        # Use existing pareto graph with both runs selected
                        run_a_id = run_a.get("slurm_job_id", "Unknown")
                        run_b_id = run_b.get("slurm_job_id", "Unknown")
                        selected_for_comparison = [run_a_id, run_b_id]
                        comparison_df = df[df["Run ID"].isin(selected_for_comparison)]

                        if not comparison_df.empty:
                            # Pareto comparison
                            st.markdown("**Pareto Frontier**")
                            pareto_comparison_fig = create_pareto_graph(
                                comparison_df,
                                selected_for_comparison,
                                show_cutoff=False,
                                cutoff_value=30.0,
                                show_frontier=False
                            )
                            st.plotly_chart(pareto_comparison_fig, width="stretch", key="comparison_pareto")

                            # Latency comparisons
                            st.markdown("**Latency Metrics**")
                            latency_metrics = [
                                ('Time to First Token (TTFT)', 'Mean TTFT (ms)', 'TTFT (ms)'),
                                ('Time Per Output Token (TPOT)', 'Mean TPOT (ms)', 'TPOT (ms)'),
                                ('Inter-Token Latency (ITL)', 'Mean ITL (ms)', 'ITL (ms)')
                            ]

                            for metric_name, metric_col, y_label in latency_metrics:
                                fig = create_latency_vs_concurrency_graph(
                                    comparison_df,
                                    selected_for_comparison,
                                    metric_name,
                                    metric_col,
                                    y_label
                                )
                                st.plotly_chart(fig, width="stretch", key=f"comparison_latency_{metric_name}")

                    with viz_tab2:
                        st.markdown("#### Delta Analysis (Run B - Run A)")
                        st.caption("Positive values indicate Run B is higher. For latency, negative is better. For throughput, positive is better.")

                        delta_data = get_delta_data_for_graphs(run_a, run_b)

                        if not delta_data.empty:
                            # TTFT Delta
                            if 'TTFT Delta (ms)' in delta_data.columns:
                                fig_ttft = go.Figure()
                                fig_ttft.add_trace(go.Scatter(
                                    x=delta_data['Concurrency'],
                                    y=delta_data['TTFT Delta (ms)'],
                                    mode='lines+markers',
                                    name='TTFT Delta',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=8)
                                ))
                                fig_ttft.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No change")
                                fig_ttft.update_layout(
                                    title="TTFT Delta (Run B - Run A)",
                                    xaxis_title="Concurrency",
                                    yaxis_title="Delta TTFT (ms)",
                                    height=400
                                )
                                st.plotly_chart(fig_ttft, width="stretch", key="delta_ttft")

                            # TPOT Delta
                            if 'TPOT Delta (ms)' in delta_data.columns:
                                fig_tpot = go.Figure()
                                fig_tpot.add_trace(go.Scatter(
                                    x=delta_data['Concurrency'],
                                    y=delta_data['TPOT Delta (ms)'],
                                    mode='lines+markers',
                                    name='TPOT Delta',
                                    line=dict(color='green', width=2),
                                    marker=dict(size=8)
                                ))
                                fig_tpot.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No change")
                                fig_tpot.update_layout(
                                    title="TPOT Delta (Run B - Run A)",
                                    xaxis_title="Concurrency",
                                    yaxis_title="Delta TPOT (ms)",
                                    height=400
                                )
                                st.plotly_chart(fig_tpot, width="stretch", key="delta_tpot")

                            # Throughput Delta
                            if 'Throughput Delta (TPS)' in delta_data.columns:
                                fig_tps = go.Figure()
                                fig_tps.add_trace(go.Scatter(
                                    x=delta_data['Concurrency'],
                                    y=delta_data['Throughput Delta (TPS)'],
                                    mode='lines+markers',
                                    name='Throughput Delta',
                                    line=dict(color='red', width=2),
                                    marker=dict(size=8)
                                ))
                                fig_tps.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No change")
                                fig_tps.update_layout(
                                    title="Output Throughput Delta (Run B - Run A)",
                                    xaxis_title="Concurrency",
                                    yaxis_title="Delta Throughput (tokens/s)",
                                    height=400
                                )
                                st.plotly_chart(fig_tps, width="stretch", key="delta_throughput")
                        else:
                            st.info("No matching concurrency levels between the two runs.")
                else:
                    st.warning("No matching concurrency levels found between the two runs to compare.")


if __name__ == "__main__":
    main()
