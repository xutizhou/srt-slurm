#!/bin/bash
# Quick start script for the Benchmark Dashboard

echo "🚀 Starting Benchmark Dashboard with uv..."
echo ""

# Run the dashboard (uv will auto-install dependencies)
echo "🎉 Launching dashboard..."
echo "   The dashboard will open in your browser at http://localhost:8501"
echo ""
uv run --with streamlit --with plotly --with pandas --with numpy streamlit run app.py
