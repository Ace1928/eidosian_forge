"""
Word Forge Database module - Database management and persistence.

Uses lazy imports to avoid circular dependency with config module.
"""

# DatabaseConfig can be imported directly (doesn't cause circular import)
from .database_config import DatabaseConfig

# Lazy imports for modules that depend on config
_lazy_imports = {
    "DBManager": ".database_manager",
    "RelationshipTypeManager": ".database_manager",
    "DatabaseError": ".database_manager",
    "TermNotFoundError": ".database_manager",
    "ConnectionError": ".database_manager",
    "QueryError": ".database_manager",
    "TransactionError": ".database_manager",
}

def __getattr__(name: str):
    """Lazy import handler to break circular dependencies."""
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __package__)
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DBManager",
    "RelationshipTypeManager",
    "DatabaseError",
    "TermNotFoundError",
    "ConnectionError",
    "QueryError",
    "TransactionError",
    "DatabaseConfig",
]
