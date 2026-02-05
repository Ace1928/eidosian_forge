"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    EIDOSIAN UNIVERSAL DECORATORS                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

The @eidosian decorator - a production-quality universal decorator that combines
logging, profiling, benchmarking, and tracing with tunable capabilities.

Usage:
    # Full decoration with all features
    @eidosian(log=True, profile=True, benchmark=True, trace=True)
    def my_function(x, y):
        return x + y
    
    # Simple usage with defaults from config
    def my_function(x, y):
        return x + y
    
    # Individual decorators
    @log_call
    @profile_call
    @benchmark_call(iterations=10)
    def my_function(x, y):
        return x + y
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, Optional, TypeVar, Union, overload
from dataclasses import dataclass

from .config import get_config, EidosianConfig
from .logging import EidosianLogger, get_logger
from .profiling import Profiler, ProfileReport
from .benchmarking import Benchmark, BenchmarkResult
from .tracing import Tracer, TraceSpan
F = TypeVar("F", bound=Callable[..., Any])
@dataclass
class DecoratorResult:
    """Result from decorated function execution."""
    result: Any
    duration_ms: float
    profile_report: Optional[ProfileReport] = None
    benchmark_result: Optional[BenchmarkResult] = None
    trace_spans: Optional[list] = None
    error: Optional[Exception] = None
class EidosianDecorator:
    """
    The universal Eidosian decorator class.
    
    Provides configurable logging, profiling, benchmarking, and tracing
    for any function or method.
    """
    
    def __init__(
        self,
        # Feature flags (None = use config)
        log: Optional[bool] = None,
        profile: Optional[bool] = None,
        benchmark: Optional[bool] = None,
        trace: Optional[bool] = None,
        
        # Logging options
        log_args: Optional[bool] = None,
        log_result: Optional[bool] = None,
        log_level: str = "DEBUG",
        
        # Profiling options
        profile_top_n: int = 20,
        profile_sort_by: str = "cumulative",
        profile_save: bool = False,
        
        # Benchmarking options
        benchmark_iterations: int = 1,
        benchmark_warmup: int = 0,
        benchmark_memory: bool = False,
        benchmark_threshold_ms: Optional[float] = None,
        
        # Tracing options
        trace_args: Optional[bool] = None,
        trace_result: Optional[bool] = None,
        trace_locals: bool = False,
        
        # General options
        name: Optional[str] = None,
        logger: Optional[EidosianLogger] = None,
        config: Optional[EidosianConfig] = None,
    ):
        self.config = config or get_config()
        
        # Resolve feature flags from config if not explicitly set
        self.log = log if log is not None else self.config.logging.enabled
        self.profile = profile if profile is not None else self.config.profiling.enabled
        self.benchmark = benchmark if benchmark is not None else self.config.benchmarking.enabled
        self.trace = trace if trace is not None else self.config.tracing.enabled
        
        # Logging options
        self.log_args = log_args if log_args is not None else self.config.logging.log_args
        self.log_result = log_result if log_result is not None else self.config.logging.log_result
        self.log_level = log_level
        
        # Profiling options
        self.profile_top_n = profile_top_n
        self.profile_sort_by = profile_sort_by
        self.profile_save = profile_save
        
        # Benchmarking options
        self.benchmark_iterations = benchmark_iterations
        self.benchmark_warmup = benchmark_warmup
        self.benchmark_memory = benchmark_memory
        self.benchmark_threshold_ms = benchmark_threshold_ms or self.config.benchmarking.threshold_ms
        
        # Tracing options
        self.trace_args = trace_args if trace_args is not None else self.config.tracing.capture_args
        self.trace_result = trace_result if trace_result is not None else self.config.tracing.capture_result
        self.trace_locals = trace_locals
        
        # General
        self.name = name
        self.logger = logger
        
        # State
        self._profiler: Optional[Profiler] = None
        self._tracer: Optional[Tracer] = None
    
    def __call__(self, func: F) -> F:
        """Decorate a function."""
        # Handle mocked functions gracefully
        func_name = self.name or getattr(func, '__name__', '<unknown>')
        
        # Get or create logger
        func_module = getattr(func, '__module__', __name__)
        logger = self.logger or get_logger(func_module)
        
        # Setup profiler if needed
        if self.profile:
            self._profiler = Profiler(
                top_n=self.profile_top_n,
                sort_by=self.profile_sort_by,
                save_stats=self.profile_save,
            )
        
        # Setup tracer if needed
        if self.trace:
            self._tracer = Tracer(
                capture_args=self.trace_args,
                capture_result=self.trace_result,
                capture_locals=self.trace_locals,
            )
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get source info only when tracing requires it
            source_file = None
            line_number = None
            if self.trace:
                try:
                    source_file = inspect.getsourcefile(func)
                    source_lines = inspect.getsourcelines(func)
                    line_number = source_lines[1] if source_lines else None
                except (TypeError, OSError):
                    source_file = None
                    line_number = None
            
            # Log entry
            if self.log:
                if self.log_args:
                    logger.function_entry(
                        func_name,
                        args=args,
                        kwargs=kwargs,
                        max_length=self.config.logging.max_arg_length,
                    )
                else:
                    logger.debug(f"→ {func_name}()")
            
            # Start tracing
            if self.trace and self._tracer:
                self._tracer.start_span(
                    func_name,
                    args=args if self.trace_args else None,
                    kwargs=kwargs if self.trace_args else None,
                    filename=source_file,
                    line_number=line_number,
                )
            
            # Start profiling
            if self.profile and self._profiler:
                self._profiler.start()
            
            # Time execution only when needed
            needs_timing = self.log or (self.benchmark_threshold_ms is not None)
            start_time = time.perf_counter() if needs_timing else None
            error: Optional[Exception] = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                error = e
                raise
                
            finally:
                duration_ms: float = 0.0
                if start_time is not None:
                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000
                
                # Stop profiling
                if self.profile and self._profiler:
                    profile_report = self._profiler.stop(func_name)
                    if self.log:
                        logger.debug(f"Profile: {profile_report.to_string(top_n=5)}")
                
                # End tracing
                if self.trace and self._tracer:
                    self._tracer.end_span(result=result, error=error)
                    if self.log and not error:
                        logger.debug(f"Trace:\n{self._tracer.to_string()}")
                
                # Log exit
                if self.log:
                    if error:
                        logger.function_error(func_name, error, duration_ms)
                    elif self.log_result:
                        logger.function_exit(
                            func_name,
                            result=result,
                            duration_ms=duration_ms,
                            max_length=self.config.logging.max_result_length,
                        )
                    else:
                        logger.debug(f"← {func_name} [{duration_ms:.2f}ms]")
                
                # Check benchmark threshold
                if (
                    self.benchmark_threshold_ms
                    and start_time is not None
                    and duration_ms > self.benchmark_threshold_ms
                ):
                    logger.warning(
                        f"⚠ {func_name} exceeded threshold: "
                        f"{duration_ms:.2f}ms > {self.benchmark_threshold_ms}ms"
                    )
        
        # Preserve function metadata
        wrapper.__eidosian_decorated__ = True
        wrapper.__eidosian_config__ = {
            "log": self.log,
            "profile": self.profile,
            "benchmark": self.benchmark,
            "trace": self.trace,
        }
        
        return wrapper  # type: ignore

def eidosian(
    func: Optional[F] = None,
    *,
    log: Optional[bool] = None,
    profile: Optional[bool] = None,
    benchmark: Optional[bool] = None,
    trace: Optional[bool] = None,
    **kwargs,
) -> Union[F, Callable[[F], F]]:
    """
    Universal Eidosian decorator.
    
    Can be used with or without parentheses:
    
        @eidosian
        def my_func(): ...
        
        @eidosian(log=True, profile=True)
        def my_func(): ...
    
    Args:
        log: Enable logging (default: from config)
        profile: Enable profiling (default: from config)
        benchmark: Enable benchmarking (default: from config)
        trace: Enable tracing (default: from config)
        **kwargs: Additional options passed to EidosianDecorator
    
    Returns:
        Decorated function
    """
    config = get_config()
    effective_log = config.logging.enabled if log is None else bool(log)
    effective_profile = config.profiling.enabled if profile is None else bool(profile)
    effective_benchmark = config.benchmarking.enabled if benchmark is None else bool(benchmark)
    effective_trace = config.tracing.enabled if trace is None else bool(trace)
    if not (effective_log or effective_profile or effective_benchmark or effective_trace):
        if func is not None:
            return func
        def passthrough(inner: F) -> F:
            return inner
        return passthrough

    decorator = EidosianDecorator(
        log=log,
        profile=profile,
        benchmark=benchmark,
        trace=trace,
        config=config,
        **kwargs,
    )
    
    if func is not None:
        # Called without parentheses: @eidosian
        return decorator(func)
    
    # Called with parentheses: @eidosian(...)
    return decorator
# Individual convenience decorators
def log_call(
    func: Optional[F] = None,
    *,
    log_args: bool = True,
    log_result: bool = True,
    level: str = "DEBUG",
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for logging function calls.
    
    Usage:
        @log_call
        def my_func(): ...
        
        @log_call(log_args=False)
        def my_func(): ...
    """
    return eidosian(
        func,
        log=True,
        profile=False,
        benchmark=False,
        trace=False,
        log_args=log_args,
        log_result=log_result,
        log_level=level,
    )

def profile_call(
    func: Optional[F] = None,
    *,
    top_n: int = 20,
    sort_by: str = "cumulative",
    save: bool = False,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for profiling function calls.
    
    Usage:
        @profile_call
        def my_func(): ...
        
        @profile_call(top_n=10, save=True)
        def my_func(): ...
    """
    return eidosian(
        func,
        log=True,  # Need logging to see output
        profile=True,
        benchmark=False,
        trace=False,
        profile_top_n=top_n,
        profile_sort_by=sort_by,
        profile_save=save,
    )

def benchmark_call(
    func: Optional[F] = None,
    *,
    iterations: int = 1,
    warmup: int = 0,
    memory: bool = False,
    threshold_ms: Optional[float] = None,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for benchmarking function calls.
    
    Usage:
        @benchmark_call
        def my_func(): ...
        
        @benchmark_call(iterations=10, memory=True)
        def my_func(): ...
    """
    return eidosian(
        func,
        log=True,  # Need logging to see output
        profile=False,
        benchmark=True,
        trace=False,
        benchmark_iterations=iterations,
        benchmark_warmup=warmup,
        benchmark_memory=memory,
        benchmark_threshold_ms=threshold_ms,
    )

def trace_call(
    func: Optional[F] = None,
    *,
    capture_args: bool = True,
    capture_result: bool = True,
    capture_locals: bool = False,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator for tracing function calls.
    
    Usage:
        @trace_call
        def my_func(): ...
        
        @trace_call(capture_locals=True)
        def my_func(): ...
    """
    return eidosian(
        func,
        log=True,  # Need logging to see output
        profile=False,
        benchmark=False,
        trace=True,
        trace_args=capture_args,
        trace_result=capture_result,
        trace_locals=capture_locals,
    )
# Check if function is already decorated

def is_eidosian_decorated(func: Callable) -> bool:
    """Check if a function is already decorated with @eidosian."""
    return getattr(func, "__eidosian_decorated__", False)

def get_eidosian_config(func: Callable) -> Optional[dict]:
    """Get the Eidosian configuration for a decorated function."""
    return getattr(func, "__eidosian_config__", None)
