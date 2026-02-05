"""
Tests for Spatial Data Structures and Profiling.
"""

import pytest
import numpy as np
from pyparticles.physics.spatial import (
    GridConfig, morton_encode, morton_decode, compute_morton_order,
    compute_cell_densities, adaptive_cell_size, pack_positions_soa,
    compute_batch_ranges
)
from pyparticles.profiling import BenchmarkResult, ProfileSession, Benchmarker


class TestGridConfig:
    """Tests for GridConfig."""
    
    def test_from_world(self):
        cfg = GridConfig.from_world(
            world_size=100.0,
            interaction_radius=2.0,
            n_particles=10000
        )
        assert cfg.half_world == 50.0
        assert cfg.cell_size >= 2.0
        assert cfg.grid_width > 0
        assert cfg.grid_height > 0
        assert cfg.max_per_cell > 0


class TestMortonEncoding:
    """Tests for Morton/Z-order encoding."""
    
    def test_encode_decode_roundtrip(self):
        for x in [0, 1, 10, 100, 255]:
            for y in [0, 1, 10, 100, 255]:
                z = morton_encode(x, y)
                dx, dy = morton_decode(z)
                assert dx == x, f"x mismatch: {dx} != {x}"
                assert dy == y, f"y mismatch: {dy} != {y}"
    
    def test_encode_ordering(self):
        """Morton codes should order spatially close cells together."""
        # Nearby cells should have nearby codes
        z00 = morton_encode(0, 0)
        z01 = morton_encode(0, 1)
        z10 = morton_encode(1, 0)
        z11 = morton_encode(1, 1)
        
        # All four should be close together
        codes = [z00, z01, z10, z11]
        assert max(codes) - min(codes) < 10


class TestMortonOrder:
    """Tests for Morton-ordered particle processing."""
    
    def test_compute_order(self):
        pos = np.array([
            [0.0, 0.0],
            [10.0, 10.0],
            [-10.0, -10.0],
            [5.0, 5.0],
        ], dtype=np.float32)
        
        order = compute_morton_order(pos, 4, cell_size=5.0, half_world=50.0,
                                     grid_w=20, grid_h=20)
        
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}


class TestCellDensities:
    """Tests for cell density computation."""
    
    def test_compute_densities(self):
        grid_counts = np.array([
            [0, 5, 10],
            [2, 8, 0],
            [3, 0, 6],
        ], dtype=np.int32)
        
        avg, max_d, empty = compute_cell_densities(grid_counts)
        
        assert max_d == 10.0
        assert empty == 3  # Three cells with 0 particles
        assert avg == pytest.approx(34.0 / 9.0, rel=0.01)


class TestAdaptiveCellSize:
    """Tests for adaptive cell sizing."""
    
    def test_increase_when_dense(self):
        new_size = adaptive_cell_size(
            avg_density=100, max_density=200,
            current_cell_size=1.0, target_avg=10.0
        )
        assert new_size > 1.0
    
    def test_decrease_when_sparse(self):
        new_size = adaptive_cell_size(
            avg_density=1, max_density=5,
            current_cell_size=1.0, target_avg=10.0
        )
        assert new_size < 1.0
    
    def test_stable_when_optimal(self):
        new_size = adaptive_cell_size(
            avg_density=10, max_density=20,
            current_cell_size=1.0, target_avg=10.0
        )
        assert new_size == 1.0


class TestSoAPacking:
    """Tests for Structure-of-Arrays packing."""
    
    def test_pack_positions(self):
        pos = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], dtype=np.float32)
        
        x_arr, y_arr = pack_positions_soa(pos, 3)
        
        assert len(x_arr) == 3
        assert len(y_arr) == 3
        assert x_arr[0] == 1.0
        assert y_arr[0] == 2.0
        assert x_arr[2] == 5.0


class TestBatchRanges:
    """Tests for batch processing utilities."""
    
    def test_compute_ranges(self):
        ranges = compute_batch_ranges(100, batch_size=32)
        
        assert ranges.shape[1] == 2  # start, end
        assert ranges[0, 0] == 0
        assert ranges[0, 1] == 32
        assert ranges[-1, 1] == 100


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""
    
    def test_to_dict(self):
        result = BenchmarkResult(
            name="test",
            n_particles=1000,
            n_types=8,
            world_size=100.0,
            mean_ms=5.0,
            std_ms=0.5,
            min_ms=4.0,
            max_ms=6.0,
            n_iterations=100,
            particles_per_sec=200000,
            interactions_per_sec=1000000,
        )
        
        d = result.to_dict()
        assert d['name'] == 'test'
        assert d['n_particles'] == 1000
        assert d['mean_ms'] == 5.0
    
    def test_repr(self):
        result = BenchmarkResult(
            name="test", n_particles=1000, n_types=8, world_size=100.0,
            mean_ms=5.0, std_ms=0.5, min_ms=4.0, max_ms=6.0,
            n_iterations=100, particles_per_sec=200000, interactions_per_sec=1000000,
        )
        
        s = repr(result)
        assert "test" in s
        assert "5.00ms" in s


class TestProfileSession:
    """Tests for ProfileSession."""
    
    def test_record_frame(self):
        session = ProfileSession()
        session.record_frame(16.67, physics_ms=10.0, render_ms=5.0)
        session.record_frame(17.0, physics_ms=11.0, render_ms=5.5)
        
        assert len(session.frame_times) == 2
        assert len(session.physics_times) == 2
        assert len(session.render_times) == 2
    
    def test_get_summary(self):
        session = ProfileSession()
        for i in range(10):
            session.record_frame(16.67 + i * 0.1, physics_ms=10.0 + i * 0.1)
        
        summary = session.get_summary()
        
        assert 'frame' in summary
        assert 'physics' in summary
        assert summary['frame']['count'] == 10
        assert summary['physics']['mean'] > 0


class TestBenchmarker:
    """Tests for Benchmarker class."""
    
    def test_create_benchmarker(self):
        bench = Benchmarker()
        assert len(bench.results) == 0
    
    def test_benchmark_physics_minimal(self):
        """Quick test with minimal particles."""
        from pyparticles.core.types import SimulationConfig
        
        cfg = SimulationConfig.small_world()
        cfg.num_particles = 100  # Very small for test speed
        
        bench = Benchmarker()
        result = bench.benchmark_physics(cfg, n_iterations=5, warmup=2)
        
        assert result.n_particles == 100
        assert result.mean_ms > 0
        assert result.particles_per_sec > 0
        assert len(bench.results) == 1
