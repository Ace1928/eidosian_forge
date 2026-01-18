"""
📚 Example code for Ollama Forge - Eidosian concept demonstrations

This package provides examples of using the Ollama Forge for various tasks,
following Eidosian principles of clarity, structure, and mathematical precision:

• Each example demonstrates a key capability of the system
• Code flows with natural elegance and purpose
• Comments reveal the deeper patterns within the implementation
• Errors are handled with grace and intelligence
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📦 Module Imports - Structured with contextual integrity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from . import (
    basic_usage,     # Core functionality demonstrations
    chat_example,    # Conversational capabilities
    generate_example, # Text generation and completion
    embedding_example,# Vector embedding operations
    quickstart,      # Rapid onboarding patterns
    version_example  # Version management utilities
)

# Update imports if needed to use ollama_forge.helpers instead of helpers
from helpers.common import (
    print_header, print_success, print_error, print_warning, print_info
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 Public API - Perfectly precise exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    "basic_usage",    # Foundational patterns and core API usage
    "chat_example",   # Interactive conversation demonstrations
    "generate_example",# Text completion and generation workflows
    "embedding_example",# Vector embeddings and semantic operations
    "quickstart",     # Zero-to-hero rapid implementation
    "version_example" # Version detection and compatibility management
]

# Self-awareness check to validate package structure
_modules_loaded = all(
    mod is not None for mod in [basic_usage, chat_example, generate_example, 
                               embedding_example, quickstart, version_example]
)

# Diagnostic information when running directly
if __name__ == "__main__":
    print(f"📘 Ollama Forge Examples: {len(__all__)} modules available")
    for name in __all__:
        module = locals().get(name)
        status = "✓" if module is not None else "✗"
        print(f"  {status} {name}")
