# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend implementations for different LLM serving frameworks.

Supported backends:
- SGLang: Full support with prefill/decode disaggregation
- vLLM: Placeholder for future support
- TensorRT-LLM: Placeholder for future support

Each backend config is a frozen dataclass that implements BackendProtocol.
"""

from .configs import (
    BackendConfig,
    BackendProtocol,
    BackendType,
    SGLangBackendConfig,
    SGLangConfig,
)

__all__ = [
    # Base types
    "BackendProtocol",
    "BackendType",
    "BackendConfig",
    # SGLang
    "SGLangBackendConfig",
    "SGLangConfig",
]
