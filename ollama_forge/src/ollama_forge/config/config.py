#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        OLLAMA FORGE CONFIGURATION                        ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║ 🌀 Contextual Integrity | 🤣 Humor as Leverage | 🎯 Exhaustive & Concise ║
# ║ ⚡ Flow Like a River    | 💡 Universal Design  | 🔄 Recursive Refinement ║
# ║ 🎭 Precision as Style   | 🚀 Velocity          | 🏛️ Structure as Control ║
# ║ 👁️ Self-Awareness       | 🔧 Elegant Solutions | 🧩 Perfect Integration  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

"""
Configuration module for Ollama Forge.

This module serves as the central source of truth for all configuration
parameters and versioning information used throughout the package.
It embodies the Eidosian principles of Structure as Control and Contextual Integrity.

Every constant, function, and docstring here stands on its own yet interlocks
seamlessly with the larger system—a fractal reflection of purpose and clarity.
"""

import datetime
import os
import platform
import sys
import time
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast

# Type definitions for better flow and precision 🎯
T = TypeVar("T")  # Generic type variable for flexible function signatures
F = TypeVar("F", bound=Callable[..., Any])  # Function type for decorators


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🖥️ Platform-Specific Configurations - Adaptive to Host Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM = platform.system().lower()
IS_WINDOWS = SYSTEM == "windows"
IS_MACOS = SYSTEM == "darwin"
IS_LINUX = SYSTEM == "linux"
IS_ARM64 = platform.machine().lower() == "arm64"
IS_X86_64 = platform.machine().lower() == "x86_64"
CPU_COUNT = os.cpu_count() or 2  # Dynamic CPU detection with fallback


# Fix getattr lambda type issues with explicit type annotations
def safe_sysconf(name: Union[int, str], default: int = 4096) -> int:
    """
    Safely get system configuration values with proper fallback. 🔒
    Accepts both string constants and integer values for maximum flexibility.
    """
    try:
        # Convert string constants to their corresponding integer values
        if isinstance(name, str):
            if hasattr(os, "sysconf_names"):
                name = os.sysconf_names.get(
                    name, -1
                )  # Get proper integer value or fallback to -1
            else:
                return default  # Windows systems don't have sysconf_names

        return os.sysconf(name) if name >= 0 else default
    except (AttributeError, ValueError, OSError):
        return default


# Calculate memory with proper typing - strings converted to proper constants
MEMORY_MB = (
    (safe_sysconf("SC_PAGE_SIZE") * safe_sysconf("SC_PHYS_PAGES") // (1024 * 1024))
    if not IS_WINDOWS
    else 4096
)

# Detect containerization - different strategies for different environments
IS_CONTAINER = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
IS_CI_ENV = any(
    env in os.environ for env in ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "TRAVIS"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📦 Package Metadata - Single Source of Truth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PACKAGE_NAME = "Ollama Forge"
PACKAGE_NAME_NORMALIZED = "ollama_forge"
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 9
VERSION = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
VERSION_RELEASE_DATE = "2025-01-15"
BUILD_TIMESTAMP = int(time.time())  # Dynamic build timestamp
PACKAGE_BIRTHDAY = 1704067200  # The moment of birth (2024-01-01) never changes

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🌐 API Configuration - Network Communication Parameters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT_OLLAMA_API_URL = "http://localhost:11434"


# Calculate timeout once, using a function to avoid constant redefinition
def calculate_timeout() -> int:
    """Calculate optimal timeout based on system specs - adaptive intelligence! ⚡"""
    base_timeout = max(30, min(120, CPU_COUNT * 15))
    return base_timeout // 2 if IS_CI_ENV else base_timeout


DEFAULT_TIMEOUT = calculate_timeout()

# More retries on less stable platforms - Windows paths are wild adventures! 🧭
DEFAULT_MAX_RETRIES = 3 if IS_WINDOWS else 4

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🤖 Model Defaults - Balance of Performance and Quality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT_CHAT_MODEL = "deepseek-r1:1.5b"  # Optimal balance of speed and quality
BACKUP_CHAT_MODEL = "qwen2.5:0.5b-Instruct"  # Excellent small model fallback
DEFAULT_EMBEDDING_MODEL = (
    DEFAULT_CHAT_MODEL  # Using chat model for embeddings improves context
)
BACKUP_EMBEDDING_MODEL = BACKUP_CHAT_MODEL

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📏 Context Window Configurations - Memory Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT_MIN_CONTEXT = 2048


# Calculate context size using a function to avoid constant redefinition
def calculate_context_size() -> int:
    """Calculate optimal context size - more RAM = bigger thoughts! 🧠"""
    base_context = min(8192, max(2048, MEMORY_MB // 512)) if MEMORY_MB > 0 else 4096
    return base_context // 2 if IS_CONTAINER or MEMORY_MB < 4096 else base_context


RECOMMENDED_CONTEXT = calculate_context_size()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 👥 Package Authors - Credit Where Due
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTHORS = [
    {"name": "Lloyd Handyside", "email": "ace1928@gmail.com"},
    {"name": "Eidos", "email": "syntheticeidos@gmail.com"},
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 Environment Control - Runtime Configuration via Environment Variables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEBUG_MODE = os.environ.get("OLLAMA_FORGE_DEBUG") == "1"
VERBOSE_MODE = os.environ.get("OLLAMA_FORGE_VERBOSE") == "1"


# Use a function to configure log level, avoiding constant redefinition
def configure_log_level() -> str:
    """Configure optimal log level based on environment. 📊"""
    base_level = os.environ.get("OLLAMA_FORGE_LOG_LEVEL", "INFO").upper()
    return "DEBUG" if DEBUG_MODE and base_level == "INFO" else base_level


LOG_LEVEL = configure_log_level()


# Use a function for progress bar settings, avoiding constant redefinition
def configure_progress_bars() -> bool:
    """Configure progress bars for optimal user experience. 📊"""
    initial_setting = os.environ.get("OLLAMA_FORGE_NO_PROGRESS") == "1" or IS_CI_ENV
    return initial_setting or not sys.stdout.isatty()


DISABLE_PROGRESS_BARS = configure_progress_bars()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📂 User and System Paths - Data Storage Locations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Platform-specific path determination - each OS has its own way of hiding files! 🙈
def configure_user_directories() -> Dict[str, str]:
    """Configure user directories based on platform. 🗂️"""
    if IS_WINDOWS:
        config_dir = os.path.join(
            os.environ.get("APPDATA", os.path.expanduser("~")), "ollama_forge"
        )
        cache_dir = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "ollama_forge",
            "cache",
        )
        data_dir = os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "ollama_forge",
            "data",
        )
    else:
        config_dir = os.path.expanduser(os.path.join("~", ".config", "ollama_forge"))
        cache_dir = os.path.expanduser(os.path.join("~", ".cache", "ollama_forge"))
        data_dir = os.path.expanduser(
            os.path.join("~", ".local", "share", "ollama_forge")
        )

    # Override with environment variables if specified - flexibility trumps convention
    return {
        "config": os.environ.get("OLLAMA_FORGE_CONFIG_DIR", config_dir),
        "cache": os.environ.get("OLLAMA_FORGE_CACHE_DIR", cache_dir),
        "data": os.environ.get("OLLAMA_FORGE_DATA_DIR", data_dir),
    }


# Set directories once, avoiding constant redefinition
user_dirs = configure_user_directories()
USER_CONFIG_DIR = user_dirs["config"]
USER_CACHE_DIR = user_dirs["cache"]
USER_DATA_DIR = user_dirs["data"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔌 API Endpoints - RESTful Interface Paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API_ENDPOINTS = {
    "version": "/api/version",
    "generate": "/api/generate",
    "chat": "/api/chat",
    "embedding": "/api/embed",
    "models": "/api/tags",  # Correct key for listing models
    "tags": "/api/tags",  # Alias for backward compatibility
    "pull": "/api/pull",
    "push": "/api/push",
    "delete": "/api/delete",
    "copy": "/api/copy",
    "create": "/api/create",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🛡️ Error Prevention - Defensive Programming Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def validate_input(
    validator: Callable[..., bool], error_msg: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator for input validation with humor-infused error messages.

    Args:
        validator: Function returning bool indicating if input is valid
        error_msg: Optional custom error message

    Returns:
        Decorated function with validation
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not validator(*args, **kwargs):
                msg = (
                    error_msg
                    or f"Invalid input for {func.__name__}... did you feed it after midnight? 🍔🌙"
                )
                raise ValueError(msg)
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def safe_dict_get(
    d: Dict[str, T], key: str, default: Optional[T] = None
) -> Optional[T]:
    """
    Safely retrieve a value from a dict - no KeyErrors here! 🛡️

    Args:
        d: Dictionary to retrieve from
        key: Key to look up
        default: Value to return if key is missing

    Returns:
        Value from dict or default
    """
    return d.get(key, default)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🛠️ Function Getters - Consistent Access Throughout the Package
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@lru_cache(maxsize=8)
def get_version_string() -> str:
    """Return the full version string with optimal caching. ✨"""
    return VERSION


@lru_cache(maxsize=8)
def get_version_tuple() -> Tuple[int, int, int]:
    """Return version as a tuple of (major, minor, patch). 📊"""
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)


def get_release_date() -> str:
    """Return the release date of the current version. 📅"""
    return VERSION_RELEASE_DATE


def get_build_age() -> int:
    """Return the age of the build in seconds - how old is your code? 👴"""
    return int(time.time()) - BUILD_TIMESTAMP


def get_package_age() -> int:
    """Return the age of the package in days since inception. 🎂"""
    return (int(time.time()) - PACKAGE_BIRTHDAY) // 86400


@lru_cache(maxsize=1)
def get_author_string() -> str:
    """Return a formatted author string with optimal caching. 👥"""
    return ", ".join(author["name"] for author in AUTHORS)


@lru_cache(maxsize=1)
def get_email_string() -> str:
    """Return a formatted email string with optimal caching. 📧"""
    return ", ".join(author["email"] for author in AUTHORS)


# Fixed type annotation in validator
def is_string(op: Any) -> bool:
    """Check if value is a string. Simple yet essential! 📝"""
    return isinstance(op, str)


@validate_input(
    is_string,
    "Operation must be a string - not a number, not a list, not a hedgehog! 🦔",
)
def get_default_api_endpoint(operation: str) -> str:
    """
    Get the API endpoint for a specific operation.

    Args:
        operation: API operation name (e.g., 'chat', 'generate')

    Returns:
        Endpoint URL path or empty string if not found
    """
    endpoint = API_ENDPOINTS.get(operation, "")
    if not endpoint and DEBUG_MODE:
        print(f"⚠️ Warning: Unknown API operation requested: {operation}")
    return endpoint


def is_debug_mode() -> bool:
    """Check if debug mode is enabled - are we wearing our X-ray specs? 🕶️"""
    return DEBUG_MODE


def get_system_info() -> Dict[str, Any]:
    """
    Return detailed system information for optimal configuration.
    Like a digital doctor's checkup for your environment! 🩺
    """
    return {
        "system": SYSTEM,
        "is_windows": IS_WINDOWS,
        "is_macos": IS_MACOS,
        "is_linux": IS_LINUX,
        "is_arm64": IS_ARM64,
        "is_x86_64": IS_X86_64,
        "cpu_count": CPU_COUNT,
        "memory_mb": MEMORY_MB,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "container": "Yes" if IS_CONTAINER else "No",
        "ci_environment": "Yes" if IS_CI_ENV else "No",
        "timestamp": datetime.datetime.now().isoformat(),
    }


def get_optimal_batch_size() -> int:
    """
    Calculate optimal batch size based on system resources.
    Because one size definitely doesn't fit all! 📏
    """
    # Start with CPU count as base
    base = CPU_COUNT
    # Adjust for memory constraints - more memory = bigger batches
    memory_factor = max(1, min(4, MEMORY_MB // 1024))
    # Reduce in containers - play nice with neighbors
    container_factor = 0.5 if IS_CONTAINER else 1.0

    return max(1, int(base * memory_factor * container_factor))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📁 Directory Creation - Ensuring Data Storage is Available
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for directory in [USER_CONFIG_DIR, USER_CACHE_DIR, USER_DATA_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        if DEBUG_MODE:
            print(f"⚠️ Warning: Could not create directory {directory}: {e}")
            print(
                "Storage operations may fail - check permissions or find a more hospitable filesystem! 🏠"
            )
        pass  # Silent pass if directory creation fails

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚙️ Runtime Configuration - Modifiable During Execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
runtime_config: Dict[str, Any] = {
    "api_url": DEFAULT_OLLAMA_API_URL,
    "timeout": DEFAULT_TIMEOUT,
    "max_retries": DEFAULT_MAX_RETRIES,
    "chat_model": DEFAULT_CHAT_MODEL,
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "context_size": RECOMMENDED_CONTEXT,
    "batch_size": get_optimal_batch_size(),
    "last_updated": time.time(),
    "update_count": 0,  # Track how many times configs change
}


def update_runtime_config(key: str, value: Any) -> bool:
    """
    Update a runtime configuration value with change tracking.

    Args:
        key: Configuration key to update
        value: New value to set

    Returns:
        True if update successful, False if key unknown
    """
    if key in runtime_config:
        # Keep track of previous value for debugging
        old_value = runtime_config.get(key)
        runtime_config[key] = value
        runtime_config["last_updated"] = time.time()
        runtime_config["update_count"] += 1

        if DEBUG_MODE and old_value != value:
            print(f"🔄 Config '{key}' changed: {old_value} → {value}")
        return True

    if DEBUG_MODE:
        print(f"⚠️ Attempted to update unknown config key: '{key}'")
    return False


def get_runtime_config(key: str, default: Any = None) -> Any:
    """
    Get a runtime configuration value with intelligent defaults.

    Args:
        key: Configuration key to retrieve
        default: Fallback value if key doesn't exist

    Returns:
        The configuration value or default
    """
    if key not in runtime_config and DEBUG_MODE and default is not None:
        print(f"ℹ️ Using default value '{default}' for missing config key '{key}'")

    return runtime_config.get(key, default)


def reset_runtime_config() -> None:
    """
    Reset runtime configuration to optimal defaults.
    Like hitting the cosmic reset button! 🔄
    """
    runtime_config.update(
        {
            "api_url": DEFAULT_OLLAMA_API_URL,
            "timeout": DEFAULT_TIMEOUT,
            "max_retries": DEFAULT_MAX_RETRIES,
            "chat_model": DEFAULT_CHAT_MODEL,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "context_size": RECOMMENDED_CONTEXT,
            "batch_size": get_optimal_batch_size(),
            "last_updated": time.time(),
            # Preserve update count for tracking
            "update_count": runtime_config.get("update_count", 0) + 1,
        }
    )

    if DEBUG_MODE:
        print("🔁 Runtime configuration reset to defaults")


def get_config_summary() -> Dict[str, Any]:
    """
    Generate a summary of current configuration state.
    The TL;DR of your setup! 📋
    """
    return {
        "version": get_version_string(),
        "system": f"{SYSTEM.capitalize()} ({platform.machine()})",
        "resources": f"{CPU_COUNT} CPUs, {MEMORY_MB}MB memory",
        "models": {
            "chat": runtime_config["chat_model"],
            "embedding": runtime_config["embedding_model"],
        },
        "environment": "debug" if DEBUG_MODE else "production",
        "api_url": runtime_config["api_url"],
        "context_size": runtime_config["context_size"],
        "last_updated": datetime.datetime.fromtimestamp(
            runtime_config["last_updated"]
        ).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 👁️ Self-Aware Initialization - Debug Information When Needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if DEBUG_MODE:
    print(f"🔍 Ollama Forge v{VERSION} configuration loaded")
    print(f"📂 User config directory: {USER_CONFIG_DIR}")
    print(f"🔧 Default API URL: {DEFAULT_OLLAMA_API_URL}")
    print(
        f"🤖 Default models: chat={DEFAULT_CHAT_MODEL}, embedding={DEFAULT_EMBEDDING_MODEL}"
    )
    print(f"💻 System: {SYSTEM.capitalize()} ({platform.machine()}), {CPU_COUNT} CPUs")
    print(f"🧠 Recommended context: {RECOMMENDED_CONTEXT} tokens")
    print(f"📦 Optimal batch size: {get_optimal_batch_size()} items")
    print(
        f"🔮 Environment: {'container' if IS_CONTAINER else 'native'}, {'CI' if IS_CI_ENV else 'local'}"
    )
    print(f"⏱️ Package age: {get_package_age()} days")
