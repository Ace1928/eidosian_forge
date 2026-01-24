"""
⚡ Performance Benchmark Tool for Eidosian MCP

Measures and tracks performance of MCP tools and operations.
Stores results for trend analysis.

Created: 2026-01-23
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add MCP to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "eidos_mcp" / "src"))

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    times_ms: List[float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def min_ms(self) -> float:
        return min(self.times_ms)
    
    @property
    def max_ms(self) -> float:
        return max(self.times_ms)
    
    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms)
    
    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)
    
    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "stdev_ms": round(self.stdev_ms, 3),
            "timestamp": self.timestamp
        }


class Benchmark:
    """
    Performance benchmark runner for MCP tools.
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path(__file__).parent / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def run(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        iterations: int = 10,
        warmup: int = 2
    ) -> BenchmarkResult:
        """
        Run a benchmark.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            args: Function arguments
            kwargs: Function keyword arguments
            iterations: Number of iterations
            warmup: Warmup iterations (not counted)
        
        Returns:
            BenchmarkResult with timing data
        """
        kwargs = kwargs or {}
        times = []
        
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)
        
        result = BenchmarkResult(name=name, iterations=iterations, times_ms=times)
        self.results.append(result)
        return result
    
    def save(self, filename: Optional[str] = None) -> Path:
        """Save benchmark results to JSON."""
        filename = filename or f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        data = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary()
        }
        
        filepath.write_text(json.dumps(data, indent=2))
        return filepath
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        return {
            "total_benchmarks": len(self.results),
            "fastest_tool": min(self.results, key=lambda r: r.mean_ms).name,
            "slowest_tool": max(self.results, key=lambda r: r.mean_ms).name,
            "total_time_ms": sum(r.mean_ms * r.iterations for r in self.results)
        }
    
    def print_results(self) -> None:
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("⚡ BENCHMARK RESULTS")
        print("=" * 70)
        
        for r in sorted(self.results, key=lambda x: x.mean_ms):
            print(f"\n{r.name}:")
            print(f"  Mean:   {r.mean_ms:8.3f} ms")
            print(f"  Median: {r.median_ms:8.3f} ms")
            print(f"  Min:    {r.min_ms:8.3f} ms")
            print(f"  Max:    {r.max_ms:8.3f} ms")
            print(f"  StdDev: {r.stdev_ms:8.3f} ms")
        
        print("\n" + "=" * 70)
        summary = self.summary()
        print(f"Total benchmarks: {summary['total_benchmarks']}")
        print(f"Fastest: {summary['fastest_tool']}")
        print(f"Slowest: {summary['slowest_tool']}")
        print("=" * 70)


def run_mcp_benchmarks() -> Benchmark:
    """Run benchmarks on MCP tools."""
    
    # Import MCP tools
    from eidos_mcp.routers import memory, knowledge, gis, diagnostics
    from eidos_mcp.plugins import init_plugins, call_tool
    
    init_plugins()
    
    bench = Benchmark()
    
    print("🔄 Running MCP tool benchmarks...\n")
    
    # Core tools
    bench.run("memory_stats", memory.memory_stats)
    bench.run("kb_search", knowledge.kb_search, args=("test",))
    bench.run("gis_get", gis.gis_get, args=("test_key",))
    bench.run("diagnostics_ping", diagnostics.diagnostics_ping)
    
    # Plugin tools
    bench.run("web_cache_stats", lambda: call_tool("web_cache_stats"))
    bench.run("identity_status", lambda: call_tool("identity_status"))
    bench.run("task_queue_status", lambda: call_tool("task_queue_status"))
    bench.run("control_status", lambda: call_tool("control_status"))
    
    return bench


if __name__ == "__main__":
    bench = run_mcp_benchmarks()
    bench.print_results()
    
    filepath = bench.save()
    print(f"\n📊 Results saved to: {filepath}")
