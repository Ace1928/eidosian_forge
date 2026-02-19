"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         EIDOSIAN CORE LIBRARY                                  ║
║                   Universal Decorators & Utilities                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

The Eidosian Core provides universal decorators, logging, profiling, tracing,
and benchmarking utilities that can be applied consistently across all forges.

Usage:
    from eidosian_core import eidosian, EidosianLogger, configure_logging

    @eidosian(log=True, profile=True, benchmark=True, trace=True)
    def my_function(x, y):
        return x + y
"""

from .decorators import (
    eidosian,
    EidosianDecorator,
    log_call,
    profile_call,
    benchmark_call,
    trace_call,
)
from .logging import (
    EidosianLogger,
    configure_logging,
    get_logger,
    LogLevel,
)
from .profiling import (
    Profiler,
    ProfileReport,
    profile_context,
)
from .benchmarking import (
    Benchmark,
    BenchmarkResult,
    benchmark_context,
)
from .tracing import (
    Tracer,
    TraceSpan,
    trace_context,
)
from .config import (
    EidosianConfig,
    get_config,
    set_config,
)
from .ports import (
    clear_port_registry_cache,
    detect_forge_root,
    detect_registry_path,
    get_service_config,
    get_service_host,
    get_service_path,
    get_service_port,
    get_service_url,
    load_port_registry,
    should_use_registry_fallback,
)

__version__ = "0.1.0"
__all__ = [
    # Main decorator
    "eidosian",
    "EidosianDecorator",
    # Individual decorators
    "log_call",
    "profile_call", 
    "benchmark_call",
    "trace_call",
    # Logging
    "EidosianLogger",
    "configure_logging",
    "get_logger",
    "LogLevel",
    # Profiling
    "Profiler",
    "ProfileReport",
    "profile_context",
    # Benchmarking
    "Benchmark",
    "BenchmarkResult",
    "benchmark_context",
    # Tracing
    "Tracer",
    "TraceSpan",
    "trace_context",
    # Config
    "EidosianConfig",
    "get_config",
    "set_config",
    # Port registry
    "clear_port_registry_cache",
    "detect_forge_root",
    "detect_registry_path",
    "get_service_config",
    "get_service_host",
    "get_service_path",
    "get_service_port",
    "get_service_url",
    "load_port_registry",
    "should_use_registry_fallback",
]
