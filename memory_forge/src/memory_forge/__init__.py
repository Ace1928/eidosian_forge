from .core.config import MemoryConfig
from .core.interfaces import MemoryItem, MemoryType
from .core.tiered_memory import (
    MemoryNamespace,
    MemoryTier,
    TieredMemoryItem,
    TieredMemorySystem,
)

__all__ = [
    "MemoryForge",
    "MemoryBroker",
    "MemoryRetrievalEngine",
    "MemoryRetrievalDaemon",
    "DaemonConfig",
    "MemoryItem",
    "MemoryType",
    "MemoryConfig",
    "TieredMemorySystem",
    "TieredMemoryItem",
    "MemoryTier",
    "MemoryNamespace",
    "AutoContextEngine",
    "ContextSuggestion",
    "ContextWindow",
    "get_auto_context",
    "MemoryIntrospector",
    "MemoryInsight",
    "MemoryStats",
    "introspect_memory",
]


def __getattr__(name):
    if name in {"AutoContextEngine", "ContextSuggestion", "ContextWindow", "get_auto_context"}:
        from .core.auto_context import AutoContextEngine, ContextSuggestion, ContextWindow, get_auto_context

        exports = {
            "AutoContextEngine": AutoContextEngine,
            "ContextSuggestion": ContextSuggestion,
            "ContextWindow": ContextWindow,
            "get_auto_context": get_auto_context,
        }
        return exports[name]
    if name in {"MemoryInsight", "MemoryIntrospector", "MemoryStats", "introspect_memory"}:
        from .core.introspection import MemoryInsight, MemoryIntrospector, MemoryStats, introspect_memory

        exports = {
            "MemoryInsight": MemoryInsight,
            "MemoryIntrospector": MemoryIntrospector,
            "MemoryStats": MemoryStats,
            "introspect_memory": introspect_memory,
        }
        return exports[name]
    if name == "MemoryForge":
        from .core.main import MemoryForge

        return MemoryForge
    if name in {"MemoryBroker", "MemoryRetrievalEngine"}:
        from .core.memory_broker import MemoryBroker
        from .core.memory_retrieval import MemoryRetrievalEngine

        exports = {
            "MemoryBroker": MemoryBroker,
            "MemoryRetrievalEngine": MemoryRetrievalEngine,
        }
        return exports[name]
    if name in {"DaemonConfig", "MemoryRetrievalDaemon"}:
        from .core.memory_daemon import DaemonConfig, MemoryRetrievalDaemon

        exports = {
            "DaemonConfig": DaemonConfig,
            "MemoryRetrievalDaemon": MemoryRetrievalDaemon,
        }
        return exports[name]
    raise AttributeError(name)
