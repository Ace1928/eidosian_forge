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
]
