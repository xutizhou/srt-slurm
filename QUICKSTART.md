# 🚀 Quick Start Guide

## Getting Started in 3 Steps

### 1️⃣ Start the Dashboard

```bash
# Option A: Use the quick start script (recommended)
./run_dashboard.sh

# Option B: Using uv directly (auto-installs dependencies)
uv run --with streamlit --with plotly --with pandas --with numpy streamlit run app.py

# Option C: Traditional way with pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Note**: We use `uv` for fast dependency management. It's already installed and will automatically handle dependencies!

### 2️⃣ Access the Dashboard

The dashboard will automatically open in your browser at:
```
http://localhost:8501
```

### 3️⃣ Start Analyzing

The dashboard automatically detects the current directory and scans for benchmark runs!

---

## Dashboard Features

### 📈 **Pareto Graph** (Main View)
- **X-axis**: Output TPS/User - throughput per concurrent user
- **Y-axis**: Output TPS/GPU - throughput efficiency per GPU
- Interactive hover to see detailed metrics
- Compare multiple runs with different colors
- Upper-right region = better performance

### 📊 **Additional Charts**
- **Throughput vs Concurrency**: See how performance scales
- **Latency Breakdown**: Compare TTFT, TPOT, and ITL metrics

### 📋 **Data Table**
- View all metrics in sortable table format
- **Download CSV** button for export to Excel/Google Sheets
- Filter by concurrency range

### ⚙️ **Configuration Viewer**
- See deployment settings (TP/DP for Prefill/Decode)
- View hardware details (GPU type, count, memory)
- Compare configurations across runs

---

## Key Metrics Explained

| Metric | Description |
|--------|-------------|
| **Output TPS** | Total output tokens per second |
| **Output TPS/GPU** | Throughput efficiency = TPS ÷ Total GPUs |
| **Output TPS/User** | Per-user throughput = TPS ÷ Concurrency |
| **Mean TTFT** | Mean Time To First Token (milliseconds) |
| **Mean TPOT** | Mean Time Per Output Token (milliseconds) |
| **Mean ITL** | Mean Inter-Token Latency (milliseconds) |

---

## Tips for Analysis

✅ **Compare Configurations**: Select multiple runs to compare different setups

✅ **Focus on Specific Loads**: Use the concurrency slider to zoom into specific ranges

✅ **Efficiency Analysis**: Points with higher Output TPS/GPU are more efficient

✅ **User Experience**: Points with higher Output TPS/User provide better per-user throughput

✅ **Export Data**: Use the Download CSV button to share results or create presentations

---

## Directory Requirements

The dashboard expects this structure:
```
logs/
├── app.py
├── [RUN_ID_1]/
│   ├── vllm_isl_*_osl_*/
│   │   └── *.json          # Benchmark results
│   └── *_config.json       # Node configs
└── [RUN_ID_2]/
    └── ...
```

Your current directory already has this structure with 2 runs detected!

---

## Troubleshooting

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Dashboard not loading data?**
- Check that you're in the logs directory
- Verify subdirectories contain `*_isl_*_osl_*/` folders with JSON files

**Need to analyze a different directory?**
- Use the sidebar "Logs Directory Path" input to point to any directory

---

## Next Steps

1. Start the dashboard with `./run_dashboard.sh`
2. Select runs to compare from the sidebar
3. Explore different visualization tabs
4. Export data for reports using the CSV download
5. Share the interactive dashboard link with your team!

**Enjoy your analysis!** 🎉
