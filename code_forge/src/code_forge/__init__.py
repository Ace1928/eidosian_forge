"""
Code Forge - AST-based code analysis and library management.

Code Forge provides tools for:
- Python AST analysis (functions, classes, variables, logic)
- Code librarian for managing code snippets and patterns
- Code indexing for knowledge integration
- Integration with knowledge_forge for semantic code understanding
"""

from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.analyzer.code_indexer import CodeIndexer, CodeElement, index_forge_codebase
from code_forge.librarian.core import CodeLibrarian

__all__ = [
    "CodeAnalyzer",
    "CodeIndexer",
    "CodeElement",
    "index_forge_codebase",
    "CodeLibrarian",
]
