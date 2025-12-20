# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and registry for benchmark runners."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


# Path to bundled benchmark scripts
SCRIPTS_DIR = Path(__file__).parent / "scripts"


class BenchmarkRunner(Protocol):
    """Protocol that all benchmark runners must implement."""

    @property
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @property
    def script_path(self) -> str:
        """Path to the benchmark script inside the container."""
        ...

    def validate_config(self, config: SrtConfig) -> list[str]:
        """Validate that config has all required fields.

        Returns:
            List of error messages (empty if valid)
        """
        ...

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        """Build the command to run the benchmark.

        Args:
            config: Full job configuration
            runtime: Runtime context with resolved paths

        Returns:
            Command as list of strings
        """
        ...


# Registry of benchmark runners
_BENCHMARK_RUNNERS: dict[str, type[BenchmarkRunner]] = {}


def register_benchmark(name: str):
    """Decorator to register a benchmark runner class.

    Usage:
        @register_benchmark("sa-bench")
        class SABenchRunner:
            ...
    """

    def decorator(cls: type[BenchmarkRunner]) -> type[BenchmarkRunner]:
        _BENCHMARK_RUNNERS[name] = cls
        return cls

    return decorator


def get_runner(benchmark_type: str) -> BenchmarkRunner:
    """Get a runner instance for the given benchmark type.

    Args:
        benchmark_type: Type of benchmark (e.g., "sa-bench", "mmlu")

    Returns:
        Instantiated runner

    Raises:
        ValueError: If benchmark type is not registered
    """
    if benchmark_type not in _BENCHMARK_RUNNERS:
        available = ", ".join(sorted(_BENCHMARK_RUNNERS.keys()))
        raise ValueError(
            f"Unknown benchmark type: {benchmark_type}. Available: {available}"
        )
    return _BENCHMARK_RUNNERS[benchmark_type]()


def list_benchmarks() -> list[str]:
    """List all registered benchmark types."""
    return sorted(_BENCHMARK_RUNNERS.keys())
