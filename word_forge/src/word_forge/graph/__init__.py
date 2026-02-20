"""
Word Forge Graph module - Knowledge graph operations.

Exports the main graph management classes and utilities.

Uses lazy imports to avoid circular dependency with config module.
"""

from typing import TYPE_CHECKING

from .graph_analysis import GraphAnalysis

# GraphConfig can be imported directly (doesn't cause circular import)
from .graph_config import GraphConfig
from .graph_io import GraphIO

# These are safe to import directly
from .graph_layout import GraphLayout
from .worker_factory import restart_worker

# Lazy imports for modules that depend on config
_lazy_imports = {
    "GraphManager": ".graph_manager",
    "GraphBuilder": ".graph_builder",
    "GraphQuery": ".graph_query",
    "GraphVisualizer": ".graph_visualizer",
    "GraphWorker": ".graph_worker",
}


def __getattr__(name: str):
    """Lazy import handler to break circular dependencies."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name], __package__)
        cls = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GraphManager",
    "GraphBuilder",
    "GraphQuery",
    "GraphLayout",
    "GraphVisualizer",
    "GraphIO",
    "GraphAnalysis",
    "GraphConfig",
    "GraphWorker",
    "restart_worker",
]
