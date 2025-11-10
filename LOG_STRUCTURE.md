# Expected Log Structure Documentation

This document defines the expected structure of benchmark logs parsed by this dashboard. If the log format changes in the future, use this as a reference to debug issues.

## Directory Structure

```
logs/
├── app.py                          # Main dashboard
├── utils/                          # Parsing utilities
└── [RUN_ID]/                      # Benchmark run directories
    ├── *_config.json              # Node configuration files
    ├── *_<service>.err            # Service error/log files
    ├── *_<service>.out            # Service output files
    └── vllm_isl_*_osl_*/          # Benchmark result directories
        └── *.json                 # Individual test results
```

## Expected File Formats

### 1. Config JSON Files (`*_config.json`)

**Filename Pattern:** `watchtower-navy-<node>_config.json`

**Expected JSON Structure:**
```json
{
  "filename": "watchtower-navy-cn01_config.json",
  "gpu_info": {
    "count": 8,
    "gpus": [
      {
        "name": "NVIDIA H100",
        "memory_total": "80GB",
        "driver_version": "535.129.03"
      }
    ]
  },
  "config": {
    "server_args": {
      "tp_size": 8,
      "dp_size": 8,
      "attention_backend": "trtllm_mla",
      "kv_cache_dtype": "fp8_e4m3",
      "disaggregation_mode": "prefill",
      "max_total_tokens": 524288,
      "chunked_prefill_size": 131072,
      "context_length": 2200,
      "served_model_name": "deepseek-ai/DeepSeek-R1",
      // ... many more server args
    }
  },
  "environment": {
    "NCCL_DEBUG": "INFO",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "SGLANG_FORCE_SHUTDOWN": "1",
    // ... many more env vars
  }
}
```

**Type Definition:** See `NodeConfig` in `utils/config_reader.py`

**Validation:** `validate_config_structure()` checks for required keys

### 2. Error Log Files (`*.err`)

**Filename Pattern:** `<node>_<service>_<id>.err`
- Examples:
  - `watchtower-navy-cn01_prefill_w0.err`
  - `watchtower-navy-cn02_decode_w0.err`
  - `watchtower-navy-cn01_nginx.err`

**Service Types:** `prefill`, `decode`, `frontend`, `nginx`, `nats`, `etcd`

**Expected Content:**
```bash
# Command line (typically around line 50-60):
+ python3 -m dynamo.sglang \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --model-path /model/ \
  --disaggregation-mode prefill \
  --tp-size 8 \
  --dp-size 8 \
  --attention-backend trtllm_mla \
  --kv-cache-dtype fp8_e4m3 \
  --max-total-tokens 524288 \
  --chunked-prefill-size 131072 \
  # ... more flags

# Runtime logs with DP/TP/EP tags (3P1D style):
[2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18, #new-token: 16384, ...
[2025-11-04 05:32:32 DP31 TP31 EP31] Decode batch, #running-req: 7, #token: 7040, token usage: 0.00, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 0, gen throughput (token/s): 6.73, #queue-req: 0,
[2025-11-04 05:27:13 DP0 TP0 EP0] Load weight end. avail mem=75.11 GB, mem usage=107.07 GB

# Runtime logs with simple TP tags (1P4D style):
[2025-11-04 07:05:55 TP0] Using KV cache dtype: torch.float8_e4m3fn
[2025-11-04 07:07:07 TP0] Prefill batch, #new-seq: 1, #new-token: 1024, ...
```

**Parsing Functions:**
- `parse_command_line_from_err()` - Extracts service topology and explicit flags
- `parse_err_file()` - Extracts runtime metrics with DP/TP/EP tags
- `parse_prefill_batch_line()` - Parses prefill batch metrics (input throughput, inflight requests, etc.)
- `parse_decode_batch_line()` - Parses decode batch metrics (gen throughput, transfer requests, prealloc requests, etc.)
- `parse_memory_line()` - Parses memory usage and allocation

## Log Patterns

### Command Line Pattern
```regex
python.*sglang.*--([a-z0-9-]+)
```

### Service Filename Pattern
```regex
(.+?)_(prefill|decode|frontend|nginx|nats|etcd)
```

### Runtime Log Pattern
```regex
# Full format (3P1D style):
\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) DP(\d+) TP(\d+) EP(\d+)\]

# Simple format (1P4D style):
\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(\d+)\]
```

### Prefill Batch Pattern
```regex
Prefill batch, #new-seq: (\d+), #new-token: (\d+), ...
```

### Decode Batch Pattern
```regex
Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), pre-allocated usage: ([\d.]+), #prealloc-req: (\d+), #transfer-req: (\d+), gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)
```

### Memory Usage Pattern
```regex
avail mem=([\d.]+)\s*GB
mem usage=([\d.]+)\s*GB
```

## Validation & Debugging

### Enabling Debug Logging

The dashboard uses Python's `logging` module. To see validation warnings:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Validation warnings will show:
- Missing expected keys in config JSON
- Missing `.err` files
- Commands not found in `.err` files
- Unparseable filenames

### Common Issues

**Issue:** "No config files found"
- Check: Are there `*_config.json` files in the run directory?
- File pattern expected: `watchtower-navy-<node>_config.json`

**Issue:** "No .err files found"
- Check: Are there `.err` files with service types?
- Expected: `<node>_<service>_<id>.err`

**Issue:** "Found .err files but no sglang commands"
- Check: Does the `.err` file contain a line with `python` and `sglang`?
- Expected format: `+ python3 -m dynamo.sglang --flag1 ...`

**Issue:** "Config missing expected keys"
- Check: Does the config have `config`, `gpu_info`, `environment`?
- Check: Does `config.config` have `server_args`?

**Issue:** "Could not parse service type from filename"
- Check: Does filename match `<node>_<service>_<id>.err`?
- Valid services: prefill, decode, frontend, nginx, nats, etcd

## Type Definitions

See `utils/config_reader.py` for config-related TypedDict definitions:
- `NodeConfig` - Config JSON structure
- `GPUInfo` - GPU information structure
- `ServerArgs` - Server arguments (partial list)
- `ParsedCommandInfo` - Result of parsing .err files

See `utils/log_parser.py` for log-related TypedDict definitions:
- `BatchMetrics` - Batch processing metrics for both prefill and decode nodes
  - Prefill fields: new_seq, new_token, cached_token, inflight_req, input_throughput
  - Decode fields: gen_throughput, transfer_req, num_tokens, preallocated_usage
  - Common fields: timestamp, dp, tp, ep, type, running_req, queue_req, prealloc_req, token_usage
- `MemoryMetrics` - Memory usage snapshots
- `ParsedErrFile` - Complete parsed .err file structure (node_info, prefill_batches, memory_snapshots, config)

## Metric Calculations

See `utils/metrics.py` for implementation. Key derived metrics:

### Output TPS/GPU (Throughput Efficiency)
```
Output TPS/GPU = Total Output Throughput (tokens/s) / Total Number of GPUs
```
Measures how efficiently each GPU is being utilized for token generation.

### Output TPS/User (Per-User Generation Rate)
```
Output TPS/User = 1000 / Mean TPOT (ms)
```
Where TPOT (Time Per Output Token) is the average time between consecutive output tokens.
This represents the actual token generation rate experienced by each user, independent of concurrency.

## Future-Proofing

If log formats change:

1. **Check validation warnings** - They'll tell you what changed
2. **Update TypedDicts** - Add/remove fields in type definitions
3. **Update regex patterns** - If filename/log formats change
4. **Update this document** - Keep it synchronized with actual format

The codebase uses dynamic parsing where possible (e.g., all server args are read from JSON, not hardcoded), so many changes won't require code updates.
