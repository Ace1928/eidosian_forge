"""
📊 Test package for Ollama Forge - Eidosian verification framework

This package contains comprehensive tests that validate Ollama Forge functionality 
through the lens of Eidosian principles:

• Structural integrity through universal test coverage
• Mathematical precision in assertions and validations
• Recursive refinement through continuous integration
• Flow-based composition of test fixtures and scenarios
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔄 Self-adapting imports - Ensuring test functionality in any context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import logging
from typing import Dict, Any

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🛡️ Import system with fault tolerance - Precision through resilience
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Import test modules for direct access with robust error handling
test_modules: Dict[str, Any] = {}

# Try to import all test modules safely - Structure as Control
for module_name in [
    "test_client", "test_chat", "test_embedding", 
    "test_helpers", "test_coverage", "test_nexus", "conftest"
]:
    try:
        # Try relative import first (package context)
        module = __import__(f".{module_name}", globals(), locals(), ["*"], 1)
        test_modules[module_name] = module
    except ImportError:
        try:
            # Try absolute import next (direct execution context)
            module = __import__(f"ollama_forge.tests.{module_name}", fromlist=["*"])
            test_modules[module_name] = module
        except ImportError as e:
            logging.debug(f"Could not import {module_name}: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✨ Expose test classes - Self-awareness through reflection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Define classes for explicit exports with perfect type safety
TestOllamaClient = getattr(test_modules.get("test_client", {}), "TestOllamaClient", None)
TestChat = getattr(test_modules.get("test_chat", {}), "TestChat", None)
TestEmbeddings = getattr(test_modules.get("test_embedding", {}), "TestEmbeddings", None)
TestHelpers = getattr(test_modules.get("test_helpers", {}), "TestHelpers", None)
TestCodeCoverage = getattr(test_modules.get("test_coverage", {}), "TestCodeCoverage", None)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📦 Module exports - API integrity through clarity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Convert available modules to namespace attributes with precision
test_client = test_modules.get("test_client")
test_chat = test_modules.get("test_chat")
test_embedding = test_modules.get("test_embedding")
test_helpers = test_modules.get("test_helpers")
test_coverage = test_modules.get("test_coverage")
test_nexus = test_modules.get("test_nexus")
conftest = test_modules.get("conftest")

# Define export symbols explicitly
__all__ = [
    # Test classes
    "TestOllamaClient", 
    "TestChat", 
    "TestEmbeddings", 
    "TestHelpers", 
    "TestCodeCoverage",
    
    # Test modules
    "test_client", 
    "test_chat", 
    "test_embedding", 
    "test_helpers",
    "test_coverage", 
    "test_nexus", 
    "conftest"
]