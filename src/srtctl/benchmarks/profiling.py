# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiling benchmark runner for torch/nsys profiling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("profiling")
class ProfilingRunner(BenchmarkRunner):
    """Profiling benchmark runner.

    Sends /start_profile API calls to workers and generates traffic
    via sglang.bench_serving to produce profiling data.

    This benchmark is auto-selected when profiling.type is "torch" or "nsys".

    Required config fields (in profiling section):
        - profiling.isl: Input sequence length
        - profiling.osl: Output sequence length
        - profiling.concurrency: Batch size for profiling
        - profiling.prefill/decode/aggregated: Phase-specific step configs
    """

    @property
    def name(self) -> str:
        return "Profiling"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/profiling/profile.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "profiling")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        p = config.profiling

        if not p.enabled:
            errors.append("profiling.type must be 'torch' or 'nsys' for profiling benchmark")
        if p.isl is None:
            errors.append("profiling.isl is required")
        if p.osl is None:
            errors.append("profiling.osl is required")
        if p.concurrency is None:
            errors.append("profiling.concurrency is required")

        # Phase config validation is already done in SrtConfig.__post_init__
        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        p = config.profiling
        r = config.resources

        return [
            "bash",
            self.script_path,
            str(r.num_prefill),
            str(r.num_decode),
            str(r.prefill_gpus),
            str(r.decode_gpus),
            str(r.prefill_gpus + r.decode_gpus),
        ]

