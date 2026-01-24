"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    EIDOSIAN BENCHMARKING SYSTEM                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
Precise benchmarking with statistical analysis and memory tracking.
"""
from __future__ import annotations
import gc
import time
import statistics
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar
from contextlib import contextmanager
import sys
# Optional memory profiling
try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False
T = TypeVar("T")
@dataclass
class BenchmarkResult:
    """
    Result of a benchmark run with statistical analysis.
    """
    name: str
    timestamp: datetime
    iterations: int
    warmup_iterations: int
    
    # Timing stats (in seconds)
    times: List[float] = field(default_factory=list)
    mean_time: float = 0.0
    median_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    std_dev: float = 0.0
    
    # Memory stats (in bytes)
    memory_peak: Optional[int] = None
    memory_allocated: Optional[int] = None
    
    # Result
    result: Any = None
    error: Optional[str] = None
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "timing": {
                "mean_ms": self.mean_time * 1000,
                "median_ms": self.median_time * 1000,
                "min_ms": self.min_time * 1000,
                "max_ms": self.max_time * 1000,
                "std_dev_ms": self.std_dev * 1000,
                "all_times_ms": [t * 1000 for t in self.times],
            },
            "memory": {
                "peak_bytes": self.memory_peak,
                "allocated_bytes": self.memory_allocated,
            } if self.memory_peak is not None else None,
            "error": self.error,
        }
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    def to_string(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Benchmark: {self.name}",
            f"  Timestamp: {self.timestamp.isoformat()}",
            f"  Iterations: {self.iterations} (warmup: {self.warmup_iterations})",
            "",
            "  Timing:",
            f"    Mean:   {self.mean_time * 1000:>10.3f} ms",
            f"    Median: {self.median_time * 1000:>10.3f} ms",
            f"    Min:    {self.min_time * 1000:>10.3f} ms",
            f"    Max:    {self.max_time * 1000:>10.3f} ms",
            f"    StdDev: {self.std_dev * 1000:>10.3f} ms",
        ]
        
        if self.memory_peak is not None:
            lines.extend([
                "",
                "  Memory:",
                f"    Peak:      {self._format_bytes(self.memory_peak)}",
                f"    Allocated: {self._format_bytes(self.memory_allocated)}",
            ])
        
        if self.error:
            lines.extend(["", f"  Error: {self.error}"])
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_bytes(n: int) -> str:
        """Format bytes as human-readable."""
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(n) < 1024.0:
                return f"{n:>10.2f} {unit}"
            n /= 1024.0
        return f"{n:>10.2f} TB"
    def save(self, path: Path) -> None:
        """Save result to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
class Benchmark:
    """
    High-precision benchmarking with memory tracking.
    
    Features:
    - Multiple iterations with warmup
    - Statistical analysis
    - Memory tracking (optional)
    - GC control
    """
    
    def __init__(
        self,
        iterations: int = 10,
        warmup: int = 1,
        record_memory: bool = False,
        gc_collect: bool = True,
        output_file: Optional[Path] = None,
        threshold_ms: Optional[float] = None,
    ):
        self.iterations = iterations
        self.warmup = warmup
        self.record_memory = record_memory and HAS_TRACEMALLOC
        self.gc_collect = gc_collect
        self.output_file = Path(output_file) if output_file else None
        self.threshold_ms = threshold_ms
    def run(
        self,
        func: Callable[..., T],
        *args,
        name: Optional[str] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Run benchmark on a function."""
        name = name or func.__name__
        
        result = BenchmarkResult(
            name=name,
            timestamp=datetime.now(),
            iterations=self.iterations,
            warmup_iterations=self.warmup,
        )
        
        # Warmup
        for _ in range(self.warmup):
            if self.gc_collect:
                gc.collect()
            try:
                func(*args, **kwargs)
            except Exception as e:
                result.error = str(e)
                return result
        
        # Memory tracking
        if self.record_memory:
            tracemalloc.start()
        
        # Benchmark iterations
        times = []
        func_result = None
        
        for _ in range(self.iterations):
            if self.gc_collect:
                gc.collect()
            
            start = time.perf_counter()
            try:
                func_result = func(*args, **kwargs)
            except Exception as e:
                result.error = str(e)
                break
            end = time.perf_counter()
            
            times.append(end - start)
        
        # Memory stats
        if self.record_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.memory_allocated = current
            result.memory_peak = peak
        
        # Calculate stats
        if times:
            result.times = times
            result.mean_time = statistics.mean(times)
            result.median_time = statistics.median(times)
            result.min_time = min(times)
            result.max_time = max(times)
            result.std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
            result.result = func_result
        
        # Check threshold
        if self.threshold_ms and result.mean_time * 1000 > self.threshold_ms:
            result.error = f"Exceeded threshold: {result.mean_time * 1000:.2f}ms > {self.threshold_ms}ms"
        
        # Save if configured
        if self.output_file:
            result.save(self.output_file)
        
        return result
    def compare(
        self,
        funcs: Dict[str, Callable],
        *args,
        **kwargs,
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple functions."""
        results = {}
        for name, func in funcs.items():
            results[name] = self.run(func, *args, name=name, **kwargs)
        return results

@contextmanager
def benchmark_context(
    name: str = "block",
    iterations: int = 1,
    record_memory: bool = False,
    print_result: bool = True,
):
    """
    Context manager for quick benchmarking.
    
    Usage:
        with benchmark_context("my_operation") as bm:
            # code to benchmark
    """
    benchmark = Benchmark(
        iterations=iterations,
        warmup=0,
        record_memory=record_memory,
    )
    
    result = BenchmarkResult(
        name=name,
        timestamp=datetime.now(),
        iterations=1,
        warmup_iterations=0,
    )
    
    if record_memory and HAS_TRACEMALLOC:
        tracemalloc.start()
    
    gc.collect()
    start = time.perf_counter()
    
    try:
        yield result
    finally:
        end = time.perf_counter()
        
        if record_memory and HAS_TRACEMALLOC:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.memory_allocated = current
            result.memory_peak = peak
        
        elapsed = end - start
        result.times = [elapsed]
        result.mean_time = elapsed
        result.median_time = elapsed
        result.min_time = elapsed
        result.max_time = elapsed
        
        if print_result:
            print(result.to_string())
