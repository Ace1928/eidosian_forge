"""
Eidosian PyParticles V6 - Profiling & Benchmarking

Performance measurement and optimization tools:
- Benchmark runner for physics kernels
- Memory profiling
- Bottleneck identification
- Performance regression testing
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Tuple
import json
import os


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    n_particles: int
    n_types: int
    world_size: float
    
    # Timing results
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_iterations: int
    
    # Throughput
    particles_per_sec: float
    interactions_per_sec: float
    
    # Memory
    memory_mb: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'n_particles': self.n_particles,
            'n_types': self.n_types,
            'world_size': self.world_size,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'n_iterations': self.n_iterations,
            'particles_per_sec': self.particles_per_sec,
            'interactions_per_sec': self.interactions_per_sec,
            'memory_mb': self.memory_mb,
        }
    
    def __repr__(self) -> str:
        return (f"Benchmark({self.name}): {self.mean_ms:.2f}ms +/- {self.std_ms:.2f}ms "
                f"({self.particles_per_sec/1000:.1f}K particles/sec)")


@dataclass
class ProfileSession:
    """Accumulated profiling data from a session."""
    start_time: float = field(default_factory=time.time)
    frame_times: List[float] = field(default_factory=list)
    physics_times: List[float] = field(default_factory=list)
    render_times: List[float] = field(default_factory=list)
    grid_times: List[float] = field(default_factory=list)
    force_times: List[float] = field(default_factory=list)
    
    def record_frame(self, total_ms: float, physics_ms: float = 0, 
                     render_ms: float = 0, grid_ms: float = 0, force_ms: float = 0):
        """Record timing data for a frame."""
        self.frame_times.append(total_ms)
        if physics_ms > 0:
            self.physics_times.append(physics_ms)
        if render_ms > 0:
            self.render_times.append(render_ms)
        if grid_ms > 0:
            self.grid_times.append(grid_ms)
        if force_ms > 0:
            self.force_times.append(force_ms)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        def stats(arr):
            if len(arr) == 0:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
            return {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'min': np.min(arr),
                'max': np.max(arr),
                'count': len(arr),
            }
        
        return {
            'duration_sec': time.time() - self.start_time,
            'frame': stats(self.frame_times),
            'physics': stats(self.physics_times),
            'render': stats(self.render_times),
            'grid': stats(self.grid_times),
            'force': stats(self.force_times),
        }
    
    def print_report(self):
        """Print formatted profiling report."""
        s = self.get_summary()
        print("\n" + "=" * 60)
        print("EIDOSIAN PYPARTICLES V6 - PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Session Duration: {s['duration_sec']:.1f}s")
        print(f"Total Frames: {s['frame']['count']}")
        if s['frame']['count'] > 0:
            print(f"Avg FPS: {1000.0 / s['frame']['mean']:.1f}")
        print()
        
        for name, data in [('Frame Total', s['frame']),
                          ('Physics', s['physics']),
                          ('Render', s['render']),
                          ('Grid Build', s['grid']),
                          ('Force Compute', s['force'])]:
            if data['count'] > 0:
                print(f"{name:15s}: {data['mean']:6.2f}ms +/- {data['std']:5.2f}ms "
                      f"(min={data['min']:.2f}, max={data['max']:.2f})")
        print("=" * 60)


class Benchmarker:
    """
    Benchmark runner for PyParticles components.
    
    Usage:
        bench = Benchmarker()
        result = bench.benchmark_physics(config, n_iterations=100)
        bench.save_results("benchmark_results.json")
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark_physics(self, config, n_iterations: int = 100, 
                          warmup: int = 10) -> BenchmarkResult:
        """
        Benchmark physics engine step performance.
        
        Args:
            config: SimulationConfig
            n_iterations: Number of timed iterations
            warmup: Warmup iterations (not timed)
            
        Returns:
            BenchmarkResult with timing statistics
        """
        from ..physics.engine import PhysicsEngine
        
        engine = PhysicsEngine(config)
        
        # Warmup
        for _ in range(warmup):
            engine.update(config.dt)
        
        # Timed runs
        times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            engine.update(config.dt)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
        
        times = np.array(times)
        mean_ms = np.mean(times)
        
        # Calculate throughput
        particles_per_sec = config.num_particles / (mean_ms / 1000.0)
        
        # Estimate interactions (N * avg_neighbors)
        avg_neighbors = min(50, config.num_particles * 0.01)  # Rough estimate
        interactions_per_sec = config.num_particles * avg_neighbors / (mean_ms / 1000.0)
        
        result = BenchmarkResult(
            name="physics_step",
            n_particles=config.num_particles,
            n_types=config.num_types,
            world_size=config.world_size,
            mean_ms=mean_ms,
            std_ms=np.std(times),
            min_ms=np.min(times),
            max_ms=np.max(times),
            n_iterations=n_iterations,
            particles_per_sec=particles_per_sec,
            interactions_per_sec=interactions_per_sec,
        )
        
        self.results.append(result)
        return result
    
    def benchmark_grid(self, config, n_iterations: int = 100) -> BenchmarkResult:
        """Benchmark spatial grid construction."""
        from ..physics.engine import PhysicsEngine
        
        engine = PhysicsEngine(config)
        
        # Warmup
        for _ in range(10):
            engine._rebuild_grid()
        
        times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            engine._rebuild_grid()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        
        times = np.array(times)
        mean_ms = np.mean(times)
        
        result = BenchmarkResult(
            name="grid_build",
            n_particles=config.num_particles,
            n_types=config.num_types,
            world_size=config.world_size,
            mean_ms=mean_ms,
            std_ms=np.std(times),
            min_ms=np.min(times),
            max_ms=np.max(times),
            n_iterations=n_iterations,
            particles_per_sec=config.num_particles / (mean_ms / 1000.0),
            interactions_per_sec=0,
        )
        
        self.results.append(result)
        return result
    
    def benchmark_scaling(self, base_config, particle_counts: List[int],
                          n_iterations: int = 50) -> List[BenchmarkResult]:
        """
        Benchmark performance scaling with particle count.
        
        Returns list of results for each particle count.
        """
        results = []
        
        for n in particle_counts:
            cfg = type(base_config)()  # Create fresh config
            cfg.num_particles = n
            cfg.max_particles = max(n, cfg.max_particles)
            cfg.world_size = base_config.world_size
            cfg.num_types = base_config.num_types
            
            result = self.benchmark_physics(cfg, n_iterations)
            results.append(result)
            print(f"  N={n:6d}: {result.mean_ms:.2f}ms ({result.particles_per_sec/1000:.1f}K/s)")
        
        return results
    
    def save_results(self, filepath: str):
        """Save all benchmark results to JSON."""
        data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': [r.to_dict() for r in self.results]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.results)} benchmark results to {filepath}")
    
    def load_results(self, filepath: str) -> List[dict]:
        """Load benchmark results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get('results', [])
    
    def compare_to_baseline(self, baseline_path: str, tolerance: float = 0.1) -> List[str]:
        """
        Compare current results to baseline.
        
        Returns list of regressions (> tolerance slower).
        """
        baseline_data = self.load_results(baseline_path)
        baseline_map = {r['name'] + str(r['n_particles']): r for r in baseline_data}
        
        regressions = []
        for result in self.results:
            key = result.name + str(result.n_particles)
            if key in baseline_map:
                baseline = baseline_map[key]
                ratio = result.mean_ms / baseline['mean_ms']
                if ratio > 1.0 + tolerance:
                    regressions.append(
                        f"{result.name} (N={result.n_particles}): "
                        f"{result.mean_ms:.2f}ms vs baseline {baseline['mean_ms']:.2f}ms "
                        f"({(ratio-1)*100:.1f}% slower)"
                    )
        
        return regressions


def run_standard_benchmarks(output_dir: str = ".") -> None:
    """
    Run standard benchmark suite and save results.
    
    This can be run from CLI to generate baseline benchmarks.
    """
    from ..core.types import SimulationConfig
    
    print("=" * 60)
    print("EIDOSIAN PYPARTICLES V6 - STANDARD BENCHMARK SUITE")
    print("=" * 60)
    
    bench = Benchmarker()
    
    # 1. Physics step benchmarks at different scales
    print("\n[1/3] Physics Step Scaling...")
    base_cfg = SimulationConfig.default()
    particle_counts = [1000, 2000, 5000, 10000, 20000, 50000]
    bench.benchmark_scaling(base_cfg, particle_counts, n_iterations=50)
    
    # 2. Grid building benchmark
    print("\n[2/3] Grid Construction...")
    for n in [5000, 20000]:
        cfg = SimulationConfig.default()
        cfg.num_particles = n
        result = bench.benchmark_grid(cfg, n_iterations=100)
        print(f"  N={n}: {result.mean_ms:.3f}ms")
    
    # 3. World size scaling
    print("\n[3/3] World Size Scaling...")
    for ws in [50, 100, 200]:
        cfg = SimulationConfig.default()
        cfg.world_size = ws
        cfg.num_particles = 10000
        result = bench.benchmark_physics(cfg, n_iterations=30)
        print(f"  world_size={ws}: {result.mean_ms:.2f}ms")
    
    # Save results
    output_path = os.path.join(output_dir, "benchmark_results.json")
    bench.save_results(output_path)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_standard_benchmarks()
