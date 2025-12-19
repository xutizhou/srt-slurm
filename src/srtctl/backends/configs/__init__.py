# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend configuration dataclasses.

Each backend has its own frozen dataclass that implements the BackendProtocol.
"""

from .base import BackendProtocol, BackendType
from .sglang import SGLangBackendConfig, SGLangConfig

# Union type for all backend configs
BackendConfig = SGLangBackendConfig

__all__ = [
    # Base types
    "BackendProtocol",
    "BackendType",
    "BackendConfig",
    # SGLang
    "SGLangBackendConfig",
    "SGLangConfig",
]

