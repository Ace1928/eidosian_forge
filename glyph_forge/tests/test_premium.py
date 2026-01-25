"""Tests for Premium Streaming Module.

Tests the premium configuration, smart buffer, stream recorder,
and premium stream engine.
"""
import pytest
import numpy as np
import time
import tempfile
import os

from glyph_forge.streaming.premium import (
    PremiumConfig,
    SmartBuffer,
    StreamRecorder,
    PremiumStreamEngine,
    stream_premium,
)


class TestPremiumConfig:
    """Test PremiumConfig dataclass."""
    
    def test_default_values(self):
        """Default config should have premium values."""
        config = PremiumConfig()
        
        assert config.resolution == "1080p"
        assert config.target_fps == 60
        assert config.render_mode == "braille"
        assert config.color_mode == "ansi256"
        assert config.buffer_seconds == 30.0
        assert config.audio_enabled is True
    
    def test_buffer_frame_calculation(self):
        """Buffer frames should be calculated from seconds."""
        config = PremiumConfig(buffer_seconds=30.0, target_fps=60)
        
        assert config.buffer_frames == 1800
        assert config.prebuffer_frames <= config.buffer_frames
    
    def test_prebuffer_capped_at_buffer(self):
        """Prebuffer should not exceed buffer size."""
        config = PremiumConfig(
            buffer_seconds=5.0,
            prebuffer_seconds=10.0,  # Larger than buffer
            target_fps=30,
        )
        
        assert config.prebuffer_frames <= config.buffer_frames
    
    def test_resolution_normalization(self):
        """Invalid resolution should be normalized."""
        config = PremiumConfig(resolution="invalid")
        assert config.resolution == "1080p"  # Falls back to default
    
    def test_pixel_resolution(self):
        """Get pixel resolution should return correct values."""
        config = PremiumConfig(resolution="1080p")
        w, h = config.get_pixel_resolution()
        
        assert w == 1920
        assert h == 1080
    
    def test_terminal_size_braille(self):
        """Terminal size for braille should divide by 2 and 4."""
        config = PremiumConfig(resolution="1080p", render_mode="braille")
        cols, rows = config.get_terminal_size()
        
        assert cols == 960  # 1920 / 2
        assert rows == 270  # 1080 / 4
    
    def test_quality_preset_maximum(self):
        """Maximum preset should be full quality."""
        config = PremiumConfig.from_quality_preset("maximum")
        
        assert config.resolution == "1080p"
        assert config.target_fps == 60
        assert config.color_mode == "truecolor"
        assert config.render_mode == "braille"
    
    def test_quality_preset_minimal(self):
        """Minimal preset should be fast."""
        config = PremiumConfig.from_quality_preset("minimal")
        
        assert config.resolution == "360p"
        assert config.target_fps == 15
        assert config.color_mode == "none"
    
    def test_record_path_auto_generated(self):
        """Record path should be auto-generated if not provided."""
        config = PremiumConfig(record_enabled=True)
        
        assert config.record_path is not None
        assert "glyph_stream_" in config.record_path


class TestSmartBuffer:
    """Test SmartBuffer class."""
    
    def test_basic_initialization(self):
        """Buffer should initialize with correct sizes."""
        buffer = SmartBuffer(max_frames=100, prebuffer_frames=10)
        
        assert buffer.effective_max == 100
        assert buffer.effective_prebuffer == 10
        assert buffer.current_size == 0
    
    def test_stream_length_capping(self):
        """Buffer should cap at stream length if known."""
        buffer = SmartBuffer(
            max_frames=1000,
            prebuffer_frames=100,
            stream_total_frames=50,  # Short stream
        )
        
        assert buffer.effective_max == 50
        assert buffer.effective_prebuffer == 50
    
    def test_add_frame(self):
        """Adding frames should increase buffer size."""
        buffer = SmartBuffer(max_frames=10, prebuffer_frames=3)
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        buffer.add_frame(frame)
        
        assert buffer.current_size == 1
    
    def test_prebuffer_event(self):
        """Prebuffer event should trigger after enough frames."""
        buffer = SmartBuffer(max_frames=100, prebuffer_frames=5)
        
        assert not buffer.is_prebuffer_complete
        
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for _ in range(5):
            buffer.add_frame(frame)
        
        assert buffer.is_prebuffer_complete
    
    def test_get_frame(self):
        """Getting frame should reduce buffer size."""
        buffer = SmartBuffer(max_frames=10, prebuffer_frames=2)
        
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for _ in range(5):
            buffer.add_frame(frame)
        
        assert buffer.current_size == 5
        
        result = buffer.get_frame(timeout=1.0)
        
        assert result is not None
        assert buffer.current_size == 4
    
    def test_buffer_overflow(self):
        """Adding beyond max should drop oldest frames."""
        buffer = SmartBuffer(max_frames=5, prebuffer_frames=2)
        
        for i in range(10):
            frame = np.full((2, 2, 3), i, dtype=np.uint8)
            buffer.add_frame(frame)
        
        assert buffer.current_size == 5
        
        # First available frame should be frame 5 (oldest was dropped)
        result = buffer.get_frame()
        assert result is not None
        assert result[0][0, 0, 0] == 5
    
    def test_buffer_level(self):
        """Buffer level should be 0-1 proportion."""
        buffer = SmartBuffer(max_frames=10, prebuffer_frames=2)
        
        assert buffer.buffer_level == 0.0
        
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        for _ in range(5):
            buffer.add_frame(frame)
        
        assert buffer.buffer_level == 0.5
    
    def test_stream_complete(self):
        """Marking stream complete should set prebuffer event."""
        buffer = SmartBuffer(max_frames=100, prebuffer_frames=50)
        
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        buffer.add_frame(frame)  # Only 1 frame
        
        assert not buffer.is_prebuffer_complete
        
        buffer.mark_stream_complete()
        
        assert buffer.is_prebuffer_complete  # Now set
    
    def test_stats(self):
        """Get stats should return comprehensive info."""
        buffer = SmartBuffer(max_frames=10, prebuffer_frames=3)
        
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        for _ in range(5):
            buffer.add_frame(frame)
        buffer.get_frame()
        
        stats = buffer.get_stats()
        
        assert stats["current_size"] == 4
        assert stats["max_size"] == 10
        assert stats["prebuffer_target"] == 3
        assert stats["frames_received"] == 5
        assert stats["frames_consumed"] == 1


class TestStreamRecorder:
    """Test StreamRecorder class."""
    
    @pytest.fixture
    def temp_output(self):
        """Create temporary output file."""
        fd, path = tempfile.mkstemp(suffix=".avi")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_initialization(self, temp_output):
        """Recorder should initialize with correct settings."""
        recorder = StreamRecorder(
            output_path=temp_output,
            width=640,
            height=480,
            fps=30.0,
        )
        
        assert recorder.width == 640
        assert recorder.height == 480
        assert recorder.fps == 30.0
        assert recorder.frame_count == 0
    
    def test_open_and_close(self, temp_output):
        """Recorder should open and close cleanly."""
        recorder = StreamRecorder(
            output_path=temp_output,
            width=640,
            height=480,
        )
        
        opened = recorder.open()
        # Note: May fail if no video codec available
        
        recorder.close()
        assert not recorder._is_open
    
    def test_context_manager(self, temp_output):
        """Recorder should work as context manager."""
        with StreamRecorder(temp_output, 320, 240) as recorder:
            pass  # Just test open/close
    
    def test_duration_calculation(self, temp_output):
        """Duration should be calculated from frame count."""
        recorder = StreamRecorder(
            output_path=temp_output,
            width=320,
            height=240,
            fps=30.0,
        )
        recorder._frame_count = 90
        
        assert recorder.duration == 3.0  # 90 frames / 30 fps


class TestPremiumStreamEngine:
    """Test PremiumStreamEngine class."""
    
    def test_engine_creation(self):
        """Engine should initialize with default config."""
        config = PremiumConfig()
        engine = PremiumStreamEngine(config)
        
        assert engine.config.resolution == "1080p"
    
    def test_engine_with_custom_config(self):
        """Engine should accept custom config."""
        config = PremiumConfig(
            resolution="720p",
            target_fps=30,
            render_mode="gradient",
        )
        engine = PremiumStreamEngine(config)
        
        assert engine.config.resolution == "720p"
        assert engine.config.target_fps == 30
    
    def test_stop_method(self):
        """Stop should set running to false."""
        engine = PremiumStreamEngine()
        engine._running = True
        
        engine.stop()
        
        assert not engine._running


class TestQualityPresets:
    """Test quality preset configurations."""
    
    def test_all_presets_valid(self):
        """All presets should create valid configs."""
        presets = ["maximum", "high", "standard", "fast", "minimal"]
        
        for preset in presets:
            config = PremiumConfig.from_quality_preset(preset)
            assert config.resolution is not None
            assert config.target_fps > 0
    
    def test_preset_hierarchy(self):
        """Presets should form quality hierarchy."""
        maximum = PremiumConfig.from_quality_preset("maximum")
        minimal = PremiumConfig.from_quality_preset("minimal")
        
        # Maximum should have higher FPS
        assert maximum.target_fps >= minimal.target_fps
        
        # Maximum should have better color
        assert maximum.color_mode != "none"
        assert minimal.color_mode == "none"


class TestBufferEdgeCases:
    """Test buffer edge cases."""
    
    def test_empty_buffer_get(self):
        """Getting from empty buffer should return None."""
        buffer = SmartBuffer(max_frames=10, prebuffer_frames=0)
        buffer._prebuffer_complete.set()  # Skip wait
        
        result = buffer.get_frame(timeout=0.1)
        assert result is None
    
    def test_very_short_stream(self):
        """Buffer should handle streams shorter than prebuffer."""
        buffer = SmartBuffer(
            max_frames=1000,
            prebuffer_frames=100,
            stream_total_frames=5,  # Very short
        )
        
        assert buffer.effective_max == 5
        assert buffer.effective_prebuffer == 5
    
    def test_single_frame_stream(self):
        """Buffer should handle single-frame stream."""
        buffer = SmartBuffer(
            max_frames=100,
            prebuffer_frames=10,
            stream_total_frames=1,
        )
        
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        buffer.add_frame(frame)
        buffer.mark_stream_complete()
        
        assert buffer.is_prebuffer_complete
        result = buffer.get_frame()
        assert result is not None


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_invalid_render_mode(self):
        """Invalid render mode should be normalized."""
        config = PremiumConfig(render_mode="invalid_mode")
        assert config.render_mode == "braille"  # Default
    
    def test_invalid_color_mode(self):
        """Invalid color mode should be normalized."""
        config = PremiumConfig(color_mode="invalid_color")
        assert config.color_mode == "ansi256"  # Default
    
    def test_negative_buffer(self):
        """Negative buffer should work (becomes 0 frames)."""
        config = PremiumConfig(buffer_seconds=-5.0)
        assert config.buffer_frames >= 0
