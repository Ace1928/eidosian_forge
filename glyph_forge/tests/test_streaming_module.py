"""Tests for the streaming module.

Tests the high-performance streaming engine components including
extractors, processors, renderers, and the main engine.
"""
import pytest
import numpy as np
import threading
import time
from unittest.mock import MagicMock, patch


class TestStreamingTypes:
    """Tests for streaming type definitions."""
    
    def test_quality_level_enum(self):
        """Test QualityLevel enum values."""
        from src.glyph_forge.streaming.types import QualityLevel
        
        assert QualityLevel.MINIMAL == 0
        assert QualityLevel.LOW == 1
        assert QualityLevel.STANDARD == 2
        assert QualityLevel.HIGH == 3
        assert QualityLevel.MAXIMUM == 4
    
    def test_edge_detector_enum(self):
        """Test EdgeDetector enum."""
        from src.glyph_forge.streaming.types import EdgeDetector
        
        assert hasattr(EdgeDetector, 'SOBEL')
        assert hasattr(EdgeDetector, 'CANNY')
        assert hasattr(EdgeDetector, 'PREWITT')
        assert hasattr(EdgeDetector, 'SCHARR')
        assert hasattr(EdgeDetector, 'LAPLACIAN')
    
    def test_video_info(self):
        """Test VideoInfo class."""
        from src.glyph_forge.streaming.types import VideoInfo
        
        info = VideoInfo(
            url="https://example.com/video.mp4",
            title="Test Video",
            fps=30.0,
            width=1920,
            height=1080,
        )
        
        assert info.url == "https://example.com/video.mp4"
        assert info.title == "Test Video"
        assert info.fps == 30.0
        assert info.width == 1920
        assert info.height == 1080
    
    def test_render_thresholds(self):
        """Test RenderThresholds calculation."""
        from src.glyph_forge.streaming.types import RenderThresholds
        
        thresholds = RenderThresholds.from_target_fps(30.0)
        
        # 30 FPS = 33.33ms per frame
        assert thresholds.reduce_ms == pytest.approx(30.0, rel=0.1)  # 90% of budget
        assert thresholds.improve_ms == pytest.approx(20.0, rel=0.1)  # 60% of budget
    
    def test_stream_metrics(self):
        """Test StreamMetrics tracking."""
        from src.glyph_forge.streaming.types import StreamMetrics
        
        metrics = StreamMetrics()
        
        # Record some frames
        for _ in range(10):
            metrics.record_frame()
            metrics.record_render(0.01)  # 10ms render time
        
        metrics.update_fps()
        
        assert metrics.frames_processed == 10
        assert metrics.average_render_time == pytest.approx(10.0, rel=0.1)
    
    def test_render_parameters(self):
        """Test RenderParameters quality adjustment."""
        from src.glyph_forge.streaming.types import RenderParameters, QualityLevel
        
        params = RenderParameters(quality_level=QualityLevel.STANDARD)
        
        # Test quality increase
        assert params.increase_quality()
        assert params.quality_level == QualityLevel.HIGH
        
        # Test quality decrease
        assert params.decrease_quality()
        assert params.quality_level == QualityLevel.STANDARD


class TestExtractors:
    """Tests for video source extractors."""
    
    def test_youtube_url_detection(self):
        """Test YouTube URL pattern matching."""
        from src.glyph_forge.streaming.extractors import YouTubeExtractor
        
        # Valid URLs
        assert YouTubeExtractor.is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert YouTubeExtractor.is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert YouTubeExtractor.is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert YouTubeExtractor.is_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ")
        assert YouTubeExtractor.is_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        
        # Invalid URLs
        assert not YouTubeExtractor.is_youtube_url("https://vimeo.com/123456")
        assert not YouTubeExtractor.is_youtube_url("https://example.com/video.mp4")
        assert not YouTubeExtractor.is_youtube_url("/path/to/video.mp4")
    
    def test_youtube_video_id_extraction(self):
        """Test YouTube video ID extraction."""
        from src.glyph_forge.streaming.extractors import YouTubeExtractor
        
        assert YouTubeExtractor.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert YouTubeExtractor.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert YouTubeExtractor.extract_video_id("https://invalid.com/video") is None
    
    def test_extraction_result(self):
        """Test ExtractionResult properties."""
        from src.glyph_forge.streaming.extractors import ExtractionResult
        
        result = ExtractionResult(
            video_url="https://example.com/video.mp4",
            audio_url="https://example.com/audio.m4a",
        )
        
        assert result.has_video
        assert result.has_audio
        
        result_no_audio = ExtractionResult(video_url="https://example.com/video.mp4")
        assert result_no_audio.has_video
        assert not result_no_audio.has_audio
    
    def test_stream_extraction_error(self):
        """Test StreamExtractionError diagnostic info."""
        from src.glyph_forge.streaming.extractors import StreamExtractionError
        
        error = StreamExtractionError(
            "Test error",
            original=ValueError("Original"),
            category="test",
        )
        
        info = error.get_diagnostic_info()
        assert info["message"] == "Test error"
        assert info["category"] == "test"
        assert info["original_type"] == "ValueError"
    
    def test_dependency_error(self):
        """Test DependencyError installation instructions."""
        from src.glyph_forge.streaming.extractors import DependencyError
        
        error = DependencyError(
            "opencv-python",
            "pip install opencv-python",
            "video streaming",
        )
        
        instructions = error.get_installation_instructions()
        assert "opencv-python" in instructions
        assert "pip install" in instructions


class TestProcessors:
    """Tests for frame processing."""
    
    def test_rgb_to_gray(self):
        """Test RGB to grayscale conversion."""
        from src.glyph_forge.streaming.processors import rgb_to_gray
        
        # Create test RGB image
        rgb = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ], dtype=np.uint8)
        
        gray = rgb_to_gray(rgb)
        
        assert gray.shape == (2, 2)
        assert gray.dtype == np.uint8
        assert gray[1, 1] >= 254  # White stays nearly white (rounding)
    
    def test_detect_edges_sobel(self):
        """Test Sobel edge detection."""
        from src.glyph_forge.streaming.processors import detect_edges
        
        # Create test image with vertical edge
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[:, 50:] = 255
        
        result = detect_edges(gray, algorithm="sobel")
        
        assert "magnitude" in result
        assert "gradient_x" in result
        assert "gradient_y" in result
        assert "direction" in result
        assert result["magnitude"].shape == gray.shape
    
    def test_detect_edges_canny(self):
        """Test Canny edge detection."""
        from src.glyph_forge.streaming.processors import detect_edges
        
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[:, 50:] = 255
        
        result = detect_edges(gray, algorithm="canny", threshold=50)
        
        assert "magnitude" in result
        assert result["magnitude"].shape == gray.shape
    
    def test_frame_processor_initialization(self):
        """Test FrameProcessor initialization."""
        from src.glyph_forge.streaming.processors import FrameProcessor
        
        processor = FrameProcessor(
            scale_factor=2,
            block_width=4,
            block_height=8,
            algorithm="sobel",
            color_enabled=True,
        )
        
        assert processor.scale_factor == 2
        assert processor.block_width == 4
        assert processor.block_height == 8
        assert processor.algorithm == "sobel"
        assert processor.color_enabled
        
        processor.shutdown()
    
    def test_frame_processor_process_frame(self):
        """Test frame processing."""
        from src.glyph_forge.streaming.processors import FrameProcessor
        
        processor = FrameProcessor(
            block_width=4,
            block_height=8,
            color_enabled=True,
        )
        
        # Create test frame (BGR)
        frame = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
        
        result = processor.process_frame(frame)
        
        assert "blocks" in result
        assert "colors" in result
        assert "edges" in result
        assert "width" in result
        assert "height" in result
        assert result["width"] > 0
        assert result["height"] > 0
        
        processor.shutdown()


class TestRenderers:
    """Tests for character rendering."""
    
    def test_character_maps(self):
        """Test character maps are properly defined."""
        from src.glyph_forge.streaming.renderers import CHARACTER_MAPS
        
        assert "gradients" in CHARACTER_MAPS
        assert "edges" in CHARACTER_MAPS
        assert "borders" in CHARACTER_MAPS
        
        assert "standard" in CHARACTER_MAPS["gradients"]
        assert "ascii" in CHARACTER_MAPS["gradients"]
        
        assert "horizontal" in CHARACTER_MAPS["edges"]
        assert "vertical" in CHARACTER_MAPS["edges"]
    
    def test_character_renderer_gradient(self):
        """Test CharacterRenderer gradient selection."""
        from src.glyph_forge.streaming.renderers import CharacterRenderer
        
        renderer = CharacterRenderer(gradient="standard", use_color=False)
        
        # Test density to character mapping
        char_dark = renderer._get_density_char(0.0)
        char_light = renderer._get_density_char(1.0)
        
        # Dark areas should use dense characters
        assert char_dark != char_light
    
    def test_character_renderer_render(self):
        """Test CharacterRenderer rendering."""
        from src.glyph_forge.streaming.renderers import CharacterRenderer
        
        renderer = CharacterRenderer(use_color=False)
        
        processed_data = {
            "blocks": np.array([[0.5, 0.8], [0.2, 0.9]]),
            "colors": None,
            "edges": None,
            "width": 2,
            "height": 2,
        }
        
        lines = renderer.render(processed_data, show_edges=False)
        
        assert len(lines) == 2
        assert len(lines[0]) == 2
    
    def test_frame_buffer(self):
        """Test FrameBuffer operations."""
        from src.glyph_forge.streaming.renderers import FrameBuffer
        
        buffer = FrameBuffer(capacity=10, target_fps=30.0)
        
        assert buffer.is_empty
        assert not buffer.is_full
        assert buffer.size == 0
        
        # Add frames
        for i in range(5):
            buffer.put(["line1", "line2"], i * 0.033, block=False)
        
        assert buffer.size == 5
        assert not buffer.is_empty
        
        # Get frame
        frame = buffer.get(block=False)
        assert frame is not None
        lines, timestamp = frame
        assert lines == ["line1", "line2"]
        
        buffer.clear()
        assert buffer.is_empty
    
    def test_frame_buffer_threading(self):
        """Test FrameBuffer thread safety."""
        from src.glyph_forge.streaming.renderers import FrameBuffer
        
        buffer = FrameBuffer(capacity=100)
        frames_added = []
        frames_retrieved = []
        
        def producer():
            for i in range(50):
                buffer.put([f"frame_{i}"], i * 0.033)
                frames_added.append(i)
        
        def consumer():
            for _ in range(50):
                frame = buffer.get(timeout=1.0)
                if frame:
                    frames_retrieved.append(frame)
        
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        consumer_thread.join()
        
        assert len(frames_added) == 50
        assert len(frames_retrieved) == 50


class TestStreamConfig:
    """Tests for stream configuration."""
    
    def test_stream_config_defaults(self):
        """Test StreamConfig default values."""
        from src.glyph_forge.streaming.engine import StreamConfig
        from src.glyph_forge.streaming.types import QualityLevel
        
        config = StreamConfig()
        
        assert config.scale_factor == 1
        assert config.block_width == 2
        assert config.block_height == 4
        assert config.quality_level == QualityLevel.STANDARD
        assert config.color_enabled is True
        assert config.adaptive_quality is False  # Disabled by default
        assert config.buffer_size == 60
        assert config.prebuffer_frames == 30
    
    def test_stream_config_validation(self):
        """Test StreamConfig validation."""
        from src.glyph_forge.streaming.engine import StreamConfig
        
        # Scale factor should be clamped to 1-4
        config = StreamConfig(scale_factor=10)
        assert config.scale_factor == 4
        
        config = StreamConfig(scale_factor=0)
        assert config.scale_factor == 1
        
        # Buffer size should be clamped
        config = StreamConfig(buffer_size=1)
        assert config.buffer_size == 10
        
        config = StreamConfig(buffer_size=1000)
        assert config.buffer_size == 300


class TestStreamEngine:
    """Tests for stream engine."""
    
    def test_stream_engine_initialization(self):
        """Test StreamEngine initialization."""
        from src.glyph_forge.streaming.engine import StreamEngine, StreamConfig
        
        config = StreamConfig(color_enabled=True, audio_enabled=False)
        engine = StreamEngine(config)
        
        assert engine.config == config
        assert not engine.is_running
        assert not engine.is_paused
    
    def test_stream_engine_with_default_config(self):
        """Test StreamEngine with default configuration."""
        from src.glyph_forge.streaming.engine import StreamEngine
        
        engine = StreamEngine()
        
        assert engine.config is not None
        assert engine.config.adaptive_quality is False


class TestAudio:
    """Tests for audio module."""
    
    def test_audio_support_check(self):
        """Test audio support detection."""
        from src.glyph_forge.streaming.audio import check_audio_support
        
        support = check_audio_support()
        
        assert "pygame" in support
        assert "simpleaudio" in support
        assert "any_available" in support
    
    def test_audio_sync(self):
        """Test AudioSync timing."""
        from src.glyph_forge.streaming.audio import AudioSync
        
        sync = AudioSync(target_fps=30.0)
        sync.start()
        
        # Simulate frames
        for _ in range(10):
            sync.frame_rendered()
        
        video_time = sync.get_video_time()
        assert video_time == pytest.approx(10 / 30.0, rel=0.1)


class TestImports:
    """Tests for module imports and public API."""
    
    def test_streaming_package_imports(self):
        """Test all public imports from streaming package."""
        from src.glyph_forge.streaming import (
            # Types
            EdgeDetector,
            GradientResult,
            QualityLevel,
            VideoInfo,
            PerformanceStats,
            TextStyle,
            RenderThresholds,
            StreamMetrics,
            RenderParameters,
            # Extractors
            YouTubeExtractor,
            VideoSourceExtractor,
            ExtractionResult,
            StreamExtractionError,
            DependencyError,
            AudioExtractor,
            # Processors
            FrameProcessor,
            supersample_image,
            rgb_to_gray,
            detect_edges,
            # Renderers
            CharacterRenderer,
            FrameRenderer,
            FrameBuffer,
            CHARACTER_MAPS,
            # Audio
            AudioPlayer,
            AudioSync,
            check_audio_support,
            # Engine
            StreamEngine,
            StreamConfig,
            stream,
            stream_youtube,
            stream_webcam,
        )
        
        # Verify all imports are not None
        assert EdgeDetector is not None
        assert StreamEngine is not None
        assert FrameProcessor is not None
    
    def test_version(self):
        """Test package version."""
        from src.glyph_forge.streaming import __version__
        
        assert __version__ == "1.0.0"
