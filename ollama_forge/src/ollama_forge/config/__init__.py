"""
Ollama Forge Configuration Package

This module exports all configuration constants, utilities, and model-related
parameters from the config and model_constants modules. It embodies the Eidosian
principles of contextual integrity and recursive refinement by providing a
single coherent interface to all configuration functionality.

Importing from this package gives you immediate access to everything you need
for configuring the Ollama Forge system - no scavenger hunt required.
"""

from typing import List

# Re-export all configuration constants and functions
from .config import (  # Platform detection; Package metadata; API configuration; Context window settings; Authors; Environment control; User paths; Utility functions; Runtime configuration
    API_ENDPOINTS,
    AUTHORS,
    BUILD_TIMESTAMP,
    CPU_COUNT,
    DEBUG_MODE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_CONTEXT,
    DEFAULT_OLLAMA_API_URL,
    DEFAULT_TIMEOUT,
    DISABLE_PROGRESS_BARS,
    IS_ARM64,
    IS_CI_ENV,
    IS_CONTAINER,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    IS_X86_64,
    LOG_LEVEL,
    MEMORY_MB,
    PACKAGE_BIRTHDAY,
    PACKAGE_NAME,
    PACKAGE_NAME_NORMALIZED,
    RECOMMENDED_CONTEXT,
    SYSTEM,
    USER_CACHE_DIR,
    USER_CONFIG_DIR,
    USER_DATA_DIR,
    VERBOSE_MODE,
    VERSION,
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
    VERSION_RELEASE_DATE,
    calculate_context_size,
    calculate_timeout,
    configure_log_level,
    configure_progress_bars,
    configure_user_directories,
    get_author_string,
    get_build_age,
    get_config_summary,
    get_default_api_endpoint,
    get_email_string,
    get_optimal_batch_size,
    get_package_age,
    get_release_date,
    get_runtime_config,
    get_system_info,
    get_version_string,
    get_version_tuple,
    is_debug_mode,
    is_string,
    reset_runtime_config,
    runtime_config,
    safe_dict_get,
    safe_sysconf,
    update_runtime_config,
    validate_input,
)

# Re-export all model constants
from .model_constants import (  # Model type definitions; Model collections and recommendations; Default and fallback models; Alias system; Utility functions
    BACKUP_CHAT_MODEL,
    BACKUP_EMBEDDING_MODEL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MODEL_ALIASES,
    MODEL_TYPE_CHAT,
    MODEL_TYPE_COMPLETION,
    MODEL_TYPE_EMBEDDING,
    RECOMMENDED_MODELS,
    get_fallback_model,
    get_model_recommendation,
    resolve_model_alias,
)

__all__: List[str] = [
    # Model types
    "MODEL_TYPE_CHAT",
    "MODEL_TYPE_COMPLETION",
    "MODEL_TYPE_EMBEDDING",
    # Model collections
    "RECOMMENDED_MODELS",
    "MODEL_ALIASES",
    # Default models
    "DEFAULT_CHAT_MODEL",
    "BACKUP_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "BACKUP_EMBEDDING_MODEL",
    # Model utility functions
    "resolve_model_alias",
    "get_fallback_model",
    "get_model_recommendation",
    # Platform detection
    "SYSTEM",
    "IS_WINDOWS",
    "IS_MACOS",
    "IS_LINUX",
    "IS_ARM64",
    "IS_X86_64",
    "CPU_COUNT",
    "MEMORY_MB",
    "IS_CONTAINER",
    "IS_CI_ENV",
    # Package metadata
    "PACKAGE_NAME",
    "PACKAGE_NAME_NORMALIZED",
    "VERSION_MAJOR",
    "VERSION_MINOR",
    "VERSION_PATCH",
    "VERSION",
    "VERSION_RELEASE_DATE",
    "BUILD_TIMESTAMP",
    "PACKAGE_BIRTHDAY",
    # API configuration
    "DEFAULT_OLLAMA_API_URL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "API_ENDPOINTS",
    # Context settings
    "DEFAULT_MIN_CONTEXT",
    "RECOMMENDED_CONTEXT",
    # Authors
    "AUTHORS",
    # Environment control
    "DEBUG_MODE",
    "VERBOSE_MODE",
    "LOG_LEVEL",
    "DISABLE_PROGRESS_BARS",
    # User paths
    "USER_CONFIG_DIR",
    "USER_CACHE_DIR",
    "USER_DATA_DIR",
    # Utility functions
    "safe_sysconf",
    "calculate_timeout",
    "calculate_context_size",
    "configure_log_level",
    "configure_progress_bars",
    "configure_user_directories",
    "validate_input",
    "safe_dict_get",
    "get_version_string",
    "get_version_tuple",
    "get_release_date",
    "get_build_age",
    "get_package_age",
    "get_author_string",
    "get_email_string",
    "is_string",
    "get_default_api_endpoint",
    "is_debug_mode",
    "get_system_info",
    "get_optimal_batch_size",
    # Runtime configuration
    "runtime_config",
    "update_runtime_config",
    "get_runtime_config",
    "reset_runtime_config",
    "get_config_summary",
]
