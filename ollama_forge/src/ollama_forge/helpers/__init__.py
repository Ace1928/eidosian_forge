"""
🌀 Ollama Forge Helpers - Eidosian architecture of utility

This package provides essential helper functions and constants for the Ollama Forge toolkit,
following the ten Eidosian principles to create a harmonious system of interdependent components:

• 🎯 Contextual Integrity - Every function and constant serves a specific purpose
• 😊 Humor as Cognitive Leverage - Error messages that inform with clarity and wit
• 🧩 Exhaustive But Concise - Complete functionality with minimal bloat
• 🌊 Flow Like a River - APIs that chain naturally and seamlessly
• 🔧 Hyper-Personal Yet Universal - Adapts to any environment while maintaining identity
• 🔄 Recursive Refinement - Functions that improve with usage
• ✨ Precision as Style - Code that is elegant because it is exact
• ⚡ Velocity as Intelligence - Fast execution through intelligent design
• 🏛️ Structure as Control - Architecture that enforces correct usage
• 👁️ Self-Awareness as Foundation - Monitoring and optimization built in
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔍 System Awareness - Adaptive configuration for any environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧩 Robust Import System - Elegant fallbacks with perfect precision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import logging

from common import (
    DEFAULT_OLLAMA_API_URL,
    async_make_api_request,
    check_ollama_installed,
    check_ollama_running,
    ensure_ollama_running,
    install_ollama,
    make_api_request,
    print_error,
    print_header,
    print_info,
    print_json,
    print_success,
    print_warning,
)
from embedding import (
    batch_calculate_similarities,
    calculate_similarity,
    normalize_vector,
    process_embeddings_response,
)

from ollama_forge.src.ollama_forge.config.model_constants import (
    BACKUP_CHAT_MODEL,
    BACKUP_EMBEDDING_MODEL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    get_fallback_model,
    get_model_recommendation,
    resolve_model_alias,
)

# Configure minimal logging - will be overridden if proper logging is configured
logging.basicConfig(level=logging.INFO)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 Public API - Perfect structure and adaptability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Define explicit exports with perfect precision - the Eidosian way
__all__ = [
    # Formatting utilities - clarity and style
    "print_header",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_json",
    # API utilities - communication and connectivity
    "make_api_request",
    "async_make_api_request",
    # Ollama management - system control
    "check_ollama_installed",
    "check_ollama_running",
    "install_ollama",
    "ensure_ollama_running",
    # Constants - foundational values
    "DEFAULT_OLLAMA_API_URL",
    "DEFAULT_CHAT_MODEL",
    "BACKUP_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "BACKUP_EMBEDDING_MODEL",
    # Model utilities - selection and transformation
    "resolve_model_alias",
    "get_fallback_model",
    "get_model_recommendation",
    # Embedding utilities - vector operations
    "calculate_similarity",
    "normalize_vector",
    "batch_calculate_similarities",
    "process_embeddings_response",
]
