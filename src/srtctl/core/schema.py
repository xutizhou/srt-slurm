#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frozen dataclass schema definitions for job configuration.

Uses marshmallow_dataclass for type-safe configuration with validation.
All config classes are frozen (immutable) after creation.

Backend configs are defined in srtctl.backends.configs/ for modularity.
"""

import itertools
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Type,
    Union,
)

import yaml
from marshmallow import Schema, ValidationError, fields
from marshmallow_dataclass import dataclass

from srtctl.backends.configs import (
    BackendConfig,
    SGLangBackendConfig,
)
from srtctl.core.formatting import (
    FormattablePath,
    FormattablePathField,
)
from srtctl.logging_utils import get_logger

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext

logger = get_logger(__name__)


# ============================================================================
# Cluster Configuration (srtslurm.yaml)
# ============================================================================


@dataclass
class ClusterConfig:
    """Cluster configuration from srtslurm.yaml."""

    default_account: Optional[str] = None
    default_partition: Optional[str] = None
    default_time_limit: Optional[str] = None
    gpus_per_node: Optional[int] = None
    network_interface: Optional[str] = None
    use_gpus_per_node_directive: bool = True
    use_segment_sbatch_directive: bool = True
    srtctl_root: Optional[str] = None
    model_paths: Optional[Dict[str, str]] = None
    containers: Optional[Dict[str, str]] = None
    cloud: Optional[Dict[str, str]] = None

    Schema: ClassVar[Type[Schema]] = Schema


# ============================================================================
# Enums
# ============================================================================


class GpuType(str, Enum):
    GB200 = "gb200"
    GB300 = "gb300"
    H100 = "h100"


class Precision(str, Enum):
    FP4 = "fp4"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


class BenchmarkType(str, Enum):
    MANUAL = "manual"
    SA_BENCH = "sa-bench"
    ROUTER = "router"
    MMLU = "mmlu"
    GPQA = "gpqa"
    LONGBENCHV2 = "longbenchv2"


class ProfilingType(str, Enum):
    NSYS = "nsys"
    TORCH = "torch"
    NONE = "none"


# ============================================================================
# Marshmallow Custom Fields
# ============================================================================


class BackendConfigField(fields.Field):
    """Marshmallow field for polymorphic backend deserialization based on type."""

    def _deserialize(
        self,
        value: Any,
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs,
    ) -> BackendConfig:
        """Deserialize backend config based on 'type' field."""
        if value is None:
            # Default to SGLang
            return SGLangBackendConfig()

        if isinstance(value, (SGLangBackendConfig)):
            return value

        if not isinstance(value, dict):
            raise ValidationError(f"Expected dict for backend config, got {type(value).__name__}")

        # Get backend type from the value dict
        backend_type = value.get("type", "sglang")

        if backend_type == "sglang":
            schema = SGLangBackendConfig.Schema()
            return schema.load(value)
        else:
            raise ValidationError(
                f"Unknown backend type: {backend_type!r}. "
                f"Supported types: sglang"
            )

    def _serialize(
        self, value: Optional[Any], attr: Optional[str], obj: Any, **kwargs
    ) -> Any:
        """Serialize backend config to dict."""
        if value is None:
            return None
        if isinstance(value, SGLangBackendConfig):
            return SGLangBackendConfig.Schema().dump(value)
        return value


class SweepConfigField(fields.Field):
    """Marshmallow field for SweepConfig."""

    def _deserialize(
        self, value: Any, attr: Optional[str], data: Optional[Mapping[str, Any]], **kwargs
    ) -> Any:
        if value is None:
            return None
        if isinstance(value, SweepConfig):
            return value
        if not isinstance(value, dict):
            raise ValidationError(f"Expected dict for sweep config, got {type(value).__name__}")

        mode = value.get("mode", "zip")
        parameters: Dict[str, List[Any]] = {}

        if "parameters" in value:
            for key, val in value["parameters"].items():
                if not isinstance(val, list):
                    raise ValidationError(f"Sweep parameter '{key}' must be a list")
                parameters[key] = val
        else:
            for key, val in value.items():
                if key == "mode":
                    continue
                if not isinstance(val, list):
                    raise ValidationError(f"Sweep parameter '{key}' must be a list")
                parameters[key] = val

        return SweepConfig(mode=mode, parameters=parameters)

    def _serialize(
        self, value: Optional[Any], attr: Optional[str], obj: Any, **kwargs
    ) -> Any:
        if value is None:
            return None
        if isinstance(value, SweepConfig):
            result: Dict[str, Any] = {"mode": value.mode}
            result.update(value.parameters)
            return result
        return value


# ============================================================================
# Sub-Configuration Dataclasses (all frozen)
# ============================================================================


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for benchmark parameter sweeps."""

    mode: Literal["zip", "grid"] = "zip"
    parameters: Dict[str, List[Any]] = field(default_factory=dict)

    def get_combinations(self) -> Iterator[Dict[str, Any]]:
        if not self.parameters:
            yield {}
            return

        if self.mode == "zip":
            param_names = list(self.parameters.keys())
            param_lists = [self.parameters[name] for name in param_names]
            for values in zip(*param_lists):
                yield dict(zip(param_names, values))
        else:
            param_names = list(self.parameters.keys())
            param_lists = [self.parameters[name] for name in param_names]
            for values in itertools.product(*param_lists):
                yield dict(zip(param_names, values))

    def __len__(self) -> int:
        if not self.parameters:
            return 1
        if self.mode == "zip":
            return len(next(iter(self.parameters.values())))
        result = 1
        for param_list in self.parameters.values():
            result *= len(param_list)
        return result

    Schema: ClassVar[Type[Schema]] = Schema


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    path: str
    container: str
    precision: str

    Schema: ClassVar[Type[Schema]] = Schema


@dataclass(frozen=True)
class ResourceConfig:
    """Resource allocation configuration."""

    gpu_type: str
    gpus_per_node: int = 4

    # Disaggregated mode
    prefill_nodes: Optional[int] = None
    decode_nodes: Optional[int] = None
    prefill_workers: Optional[int] = None
    decode_workers: Optional[int] = None

    # Aggregated mode
    agg_nodes: Optional[int] = None
    agg_workers: Optional[int] = None

    @property
    def is_disaggregated(self) -> bool:
        return self.prefill_nodes is not None or self.decode_nodes is not None

    @property
    def total_nodes(self) -> int:
        if self.is_disaggregated:
            return (self.prefill_nodes or 0) + (self.decode_nodes or 0)
        return self.agg_nodes or 1

    @property
    def num_prefill(self) -> int:
        return self.prefill_workers or 0

    @property
    def num_decode(self) -> int:
        return self.decode_workers or 0

    @property
    def num_agg(self) -> int:
        return self.agg_workers or 0

    @property
    def gpus_per_prefill(self) -> int:
        if self.prefill_nodes and self.prefill_workers:
            return (self.prefill_nodes * self.gpus_per_node) // self.prefill_workers
        return self.gpus_per_node

    @property
    def gpus_per_decode(self) -> int:
        if self.decode_nodes and self.decode_workers:
            return (self.decode_nodes * self.gpus_per_node) // self.decode_workers
        return self.gpus_per_node

    @property
    def gpus_per_agg(self) -> int:
        if self.agg_nodes and self.agg_workers:
            return (self.agg_nodes * self.gpus_per_node) // self.agg_workers
        return self.gpus_per_node

    Schema: ClassVar[Type[Schema]] = Schema


@dataclass(frozen=True)
class SlurmConfig:
    """SLURM job settings."""

    account: Optional[str] = None
    partition: Optional[str] = None
    time_limit: Optional[str] = None

    Schema: ClassVar[Type[Schema]] = Schema


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark configuration."""

    type: str = "manual"
    isl: Optional[int] = None
    osl: Optional[int] = None
    concurrencies: Optional[Union[List[int], str]] = None
    req_rate: Optional[str] = "inf"
    sweep: Optional[Annotated[SweepConfig, SweepConfigField()]] = None
    num_examples: Optional[int] = None
    max_tokens: Optional[int] = None
    repeat: Optional[int] = None
    num_threads: Optional[int] = None
    max_context_length: Optional[int] = None
    categories: Optional[List[str]] = None

    def get_concurrency_list(self) -> List[int]:
        if self.concurrencies is None:
            return []
        if isinstance(self.concurrencies, str):
            return [int(x) for x in self.concurrencies.split("x")]
        return list(self.concurrencies)

    Schema: ClassVar[Type[Schema]] = Schema


@dataclass(frozen=True)
class ProfilingConfig:
    """Profiling configuration."""

    type: str = "none"
    isl: Optional[int] = None
    osl: Optional[int] = None
    concurrency: Optional[int] = None
    start_step: Optional[int] = None
    stop_step: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.type != "none"

    Schema: ClassVar[Type[Schema]] = Schema


@dataclass(frozen=True)
class FrontendConfig:
    """Frontend/router configuration."""

    use_sglang_router: bool = False
    enable_multiple_frontends: bool = True
    num_additional_frontends: int = 9
    sglang_router_args: Optional[Dict[str, Any]] = None
    dynamo_frontend_args: Optional[Dict[str, Any]] = None

    def get_router_args_list(self) -> List[str]:
        args = self.sglang_router_args if self.use_sglang_router else self.dynamo_frontend_args
        if not args:
            return []
        result = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key}")
            elif value is not False and value is not None:
                result.extend([f"--{key}", str(value)])
        return result

    Schema: ClassVar[Type[Schema]] = Schema


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration with formattable paths."""

    log_dir: Annotated[FormattablePath, FormattablePathField()] = field(
        default_factory=lambda: FormattablePath(template="./outputs/{job_id}/logs")
    )
    results_dir: Optional[Annotated[FormattablePath, FormattablePathField()]] = None

    Schema: ClassVar[Type[Schema]] = Schema


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration."""

    max_attempts: int = 60
    interval_seconds: int = 10

    Schema: ClassVar[Type[Schema]] = Schema


# ============================================================================
# Main Configuration Dataclass
# ============================================================================


@dataclass(frozen=True)
class SrtConfig:
    """Complete srtctl job configuration (frozen, immutable).

    This is the main configuration type returned by load_config().

    The backend field supports polymorphic deserialization:
    - type: sglang -> SGLangBackendConfig
    """

    name: str
    model: ModelConfig
    resources: ResourceConfig

    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    backend: Annotated[BackendConfig, BackendConfigField()] = field(
        default_factory=SGLangBackendConfig
    )
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)

    environment: Dict[str, str] = field(default_factory=dict)
    container_mounts: Dict[
        Annotated[FormattablePath, FormattablePathField()],
        Annotated[FormattablePath, FormattablePathField()],
    ] = field(default_factory=dict)
    extra_mount: Optional[tuple[str, ...]] = None
    srun_options: Dict[str, str] = field(default_factory=dict)
    sbatch_directives: Dict[str, str] = field(default_factory=dict)
    enable_config_dump: bool = True

    # Custom setup script (runs before dynamo install and worker startup)
    # e.g. "custom-setup.sh" -> runs /configs/custom-setup.sh
    setup_script: Optional[str] = None

    Schema: ClassVar[Type[Schema]] = Schema

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "SrtConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        schema = cls.Schema()
        return schema.load(data)

    @property
    def served_model_name(self) -> str:
        """Get the served model name from backend config or model path."""
        # Try SGLang-specific extraction
        if isinstance(self.backend, SGLangBackendConfig) and self.backend.sglang_config:
            for cfg in [
                self.backend.sglang_config.prefill,
                self.backend.sglang_config.aggregated,
            ]:
                if cfg:
                    name = cfg.get("served-model-name") or cfg.get("served_model_name")
                    if name:
                        return name
        # Fallback to model path basename
        return Path(self.model.path).name

    @property
    def backend_type(self) -> str:
        """Get the backend type string."""
        return self.backend.type
