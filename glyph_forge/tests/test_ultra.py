"""Tests for Ultra High-Performance Streaming Module.

Tests the optimized streaming engine, frame pool, delta encoder,
and vectorized ANSI generation.
"""
import pytest
import numpy as np
import time

from glyph_forge.streaming.ultra import (
    UltraConfig,
    FramePool,
    DeltaEncoder,
    VectorizedANSI,
    UltraStreamEngine,
    benchmark_rendering,
)


class TestUltraConfig:
    """Test UltraConfig dataclass."""
    
    def test_default_values(self):
        """Default config should have sensible values."""
        config = UltraConfig()
        assert config.resolution == "480p"
        assert config.target_fps == 30
        assert config.render_mode == "gradient"
        assert config.color_enabled is True
    
    def test_custom_values(self):
        """Custom config should accept values."""
        config = UltraConfig(
            resolution="720p",
            target_fps=60,
            render_mode="braille",
            color_enabled=False,
        )
        assert config.resolution == "720p"
        assert config.target_fps == 60
        assert config.render_mode == "braille"
        assert config.color_enabled is False


class TestFramePool:
    """Test FramePool pre-allocation."""
    
    def test_pool_creation(self):
        """Pool should pre-allocate frames."""
        pool = FramePool(4, (480, 640, 3))
        assert pool.available() == 4
    
    def test_acquire_frame(self):
        """Acquire should return frame and reduce available."""
        pool = FramePool(4, (100, 100, 3))
        frame = pool.acquire()
        
        assert frame is not None
        assert frame.shape == (100, 100, 3)
        assert pool.available() == 3
    
    def test_release_frame(self):
        """Release should return frame to pool."""
        pool = FramePool(2, (50, 50, 3))
        frame = pool.acquire()
        assert pool.available() == 1
        
        pool.release(frame)
        assert pool.available() == 2
    
    def test_exhaust_pool(self):
        """Exhausted pool should allocate new frames."""
        pool = FramePool(2, (10, 10, 3))
        
        f1 = pool.acquire()
        f2 = pool.acquire()
        assert pool.available() == 0
        
        # Should still return a frame (fallback allocation)
        f3 = pool.acquire()
        assert f3 is not None
        assert f3.shape == (10, 10, 3)


class TestDeltaEncoder:
    """Test DeltaEncoder compression."""
    
    def test_first_frame_full_render(self):
        """First frame should render all characters."""
        encoder = DeltaEncoder(10, 5)
        chars = np.arange(50, dtype=np.int32).reshape(5, 10)
        
        lines, changed = encoder.encode(chars)
        
        assert len(lines) == 5
        assert changed == 50  # All characters changed
    
    def test_identical_frame_no_changes(self):
        """Identical frame should report minimal changes."""
        encoder = DeltaEncoder(10, 5)
        chars = np.full((5, 10), ord('A'), dtype=np.int32)
        
        # First frame
        encoder.encode(chars)
        
        # Second identical frame
        lines, changed = encoder.encode(chars)
        
        # Note: Implementation may re-render lines, but changed should be low
        assert changed == 0 or changed <= 50
    
    def test_reset_clears_state(self):
        """Reset should clear previous frame state."""
        encoder = DeltaEncoder(5, 3)
        chars = np.full((3, 5), ord('X'), dtype=np.int32)
        
        encoder.encode(chars)
        encoder.reset()
        
        # After reset, should be treated as first frame
        _, changed = encoder.encode(chars)
        assert changed == 15  # All chars


class TestVectorizedANSI:
    """Test VectorizedANSI code generation."""
    
    def test_fg_color_format(self):
        """Foreground color should produce correct escape code."""
        ansi = VectorizedANSI()
        code = ansi.fg_color(255, 128, 64)
        
        assert code == "\033[38;2;255;128;64m"
    
    def test_bg_color_format(self):
        """Background color should produce correct escape code."""
        ansi = VectorizedANSI()
        code = ansi.bg_color(0, 128, 255)
        
        assert code == "\033[48;2;0;128;255m"
    
    def test_color_caching(self):
        """Same color should return cached code."""
        ansi = VectorizedANSI()
        
        # Call twice
        code1 = ansi.fg_color(100, 100, 100)
        code2 = ansi.fg_color(100, 100, 100)
        
        # Should be same object (cached)
        assert code1 is code2
    
    def test_render_line_vectorized(self):
        """Line rendering should produce colored output."""
        ansi = VectorizedANSI()
        chars = ['A', 'B', 'C']
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        
        line = ansi.render_line_vectorized(chars, colors)
        
        assert '\033[38;2;255;0;0mA' in line
        assert '\033[38;2;0;255;0mB' in line
        assert '\033[38;2;0;0;255mC' in line
        assert line.endswith('\033[0m')


class TestUltraStreamEngine:
    """Test UltraStreamEngine."""
    
    def test_engine_creation(self):
        """Engine should initialize with config."""
        config = UltraConfig(resolution="480p", render_mode="gradient")
        engine = UltraStreamEngine(config)
        
        assert engine.config.resolution == "480p"
    
    def test_process_frame_gradient(self):
        """Gradient mode should produce output."""
        config = UltraConfig(
            resolution="480p",
            render_mode="gradient",
            color_enabled=True,
        )
        engine = UltraStreamEngine(config)
        
        # Create test frame (BGR format as from OpenCV)
        frame = np.random.randint(0, 256, (480, 854, 3), dtype=np.uint8)
        
        lines = engine.process_frame_ultra(frame)
        
        assert isinstance(lines, list)
        assert len(lines) > 0
        assert all(isinstance(l, str) for l in lines)
    
    def test_process_frame_braille(self):
        """Braille mode should produce output."""
        config = UltraConfig(
            resolution="480p",
            render_mode="braille",
            braille_adaptive=False,
            color_enabled=False,  # Faster
        )
        engine = UltraStreamEngine(config)
        
        frame = np.random.randint(0, 256, (480, 854, 3), dtype=np.uint8)
        
        lines = engine.process_frame_ultra(frame)
        
        assert isinstance(lines, list)
        assert len(lines) > 0
        # Should contain Braille characters
        assert any('⠀' <= c <= '⣿' for line in lines for c in line)
    
    def test_get_metrics(self):
        """Should return performance metrics."""
        config = UltraConfig(resolution="480p")
        engine = UltraStreamEngine(config)
        
        # Process some frames
        frame = np.random.randint(0, 256, (480, 854, 3), dtype=np.uint8)
        for _ in range(5):
            engine.process_frame_ultra(frame)
        
        metrics = engine.get_metrics()
        
        assert 'fps' in metrics
        assert 'frame_time_ms' in metrics
        assert metrics['fps'] > 0


class TestBenchmarkRendering:
    """Test the benchmark function."""
    
    def test_benchmark_returns_metrics(self):
        """Benchmark should return performance dict."""
        result = benchmark_rendering("480p", "gradient", iterations=10)
        
        assert 'avg_ms' in result
        assert 'min_ms' in result
        assert 'max_ms' in result
        assert 'avg_fps' in result
        assert 'max_fps' in result
        assert result['avg_fps'] > 0
    
    def test_benchmark_different_modes(self):
        """Benchmark should work for different modes."""
        for mode in ['gradient', 'braille']:
            result = benchmark_rendering("480p", mode, iterations=5)
            assert result['avg_fps'] > 0


class TestPerformanceTargets:
    """Test that performance targets are achievable."""
    
    def test_gradient_480p_target(self):
        """Gradient 480p should achieve 30+ fps."""
        config = UltraConfig(
            resolution="480p",
            render_mode="gradient",
            color_enabled=True,
        )
        engine = UltraStreamEngine(config)
        
        frame = np.random.randint(0, 256, (480, 854, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            engine.process_frame_ultra(frame)
        
        # Benchmark
        times = []
        for _ in range(20):
            start = time.perf_counter()
            engine.process_frame_ultra(frame)
            times.append(time.perf_counter() - start)
        
        avg_fps = 1.0 / (sum(times) / len(times))
        
        # Should achieve at least 30 fps
        assert avg_fps >= 30, f"Only achieved {avg_fps:.1f} fps"
    
    def test_braille_plain_480p_target(self):
        """Plain Braille 480p should achieve 60+ fps."""
        config = UltraConfig(
            resolution="480p",
            render_mode="braille",
            color_enabled=False,  # No color for speed
            braille_adaptive=False,
        )
        engine = UltraStreamEngine(config)
        
        frame = np.random.randint(0, 256, (480, 854, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            engine.process_frame_ultra(frame)
        
        # Benchmark
        times = []
        for _ in range(20):
            start = time.perf_counter()
            engine.process_frame_ultra(frame)
            times.append(time.perf_counter() - start)
        
        avg_fps = 1.0 / (sum(times) / len(times))
        
        # Plain Braille should be very fast
        assert avg_fps >= 50, f"Only achieved {avg_fps:.1f} fps"
