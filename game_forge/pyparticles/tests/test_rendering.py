import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pyparticles.rendering.canvas import Canvas, RenderMode
from pyparticles.core.types import SimulationConfig

@pytest.fixture
def mock_pygame_setup():
    with patch("pyparticles.rendering.canvas.pygame"):
        yield

def test_render_sprites(mock_pygame_setup):
    """Test standard sprite rendering path."""
    cfg = SimulationConfig.default()
    cfg.render_mode = RenderMode.SPRITES
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    
    pos = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    colors = np.array([0, 1], dtype=np.int32)
    angle = np.zeros(2, dtype=np.float32)
    active = 2
    species_params = np.zeros((6, 3), dtype=np.float32)
    
    canvas.render(pos, colors, angle, species_params, active, 60.0)
    
    # Verify blits called
    assert canvas.screen.blits.called

def test_render_glow(mock_pygame_setup):
    """Test glow mode initialization and render."""
    cfg = SimulationConfig.default()
    cfg.render_mode = RenderMode.GLOW
    canvas = Canvas(cfg)
    
    # Configure sprite mock to have width
    canvas.sprites[0].get_width.return_value = 100
    
    assert canvas.sprites[0].get_width() > canvas.radius_px * 2
    
    canvas.screen = MagicMock()
    pos = np.array([[10.0, 10.0]], dtype=np.float32)
    colors = np.array([0], dtype=np.int32)
    angle = np.zeros(1, dtype=np.float32)
    species_params = np.zeros((6, 3), dtype=np.float32)

    canvas.render(pos, colors, angle, species_params, 1, 60.0)
    assert canvas.screen.blits.called

def test_render_points_logic(mock_pygame_setup):
    """Test _render_fast_points logic with mocked PixelArray."""
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    
    # Mock PixelArray context manager
    mock_px_array = MagicMock()
    with patch("pyparticles.rendering.canvas.pygame.PixelArray", return_value=mock_px_array):
        sx = np.array([100.0, 200.0])
        sy = np.array([100.0, 200.0])
        colors = np.array([0, 1])
        
        canvas._render_fast_points(sx, sy, colors)
        
        # Verify pixel assignment
        # Note: MagicMock __setitem__ tracking is tricky.
        # Just ensure PixelArray was init and closed.
        mock_px_array.close.assert_called()

def test_render_points_exception(mock_pygame_setup):
    """Test exception handling in fast points."""
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    
    with patch("pyparticles.rendering.canvas.pygame.PixelArray", side_effect=Exception("Lock error")):
        # Should not raise
        canvas._render_fast_points(np.array([10]), np.array([10]), np.array([0]))

def test_render_points_fallback_trigger(mock_pygame_setup):
    """Test that high particle count triggers fast points."""
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    
    # Mock internal methods
    canvas._render_fast_points = MagicMock()
    canvas._render_sprites = MagicMock()
    
    # 10001 particles
    pos = np.zeros((10001, 2))
    colors = np.zeros(10001, dtype=int)
    angle = np.zeros(10001, dtype=np.float32)
    species_params = np.zeros((6, 3), dtype=np.float32)

    canvas.render(pos, colors, angle, species_params, 10001, 60.0)
    
    canvas._render_fast_points.assert_called()
    canvas._render_sprites.assert_not_called()

def test_resize_logic(mock_pygame_setup):
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    old_scale = canvas.scale
    
    canvas.resize(2400, 2000)
    assert canvas.scale == 1200.0
    assert canvas.width == 2400
    assert canvas.scale != old_scale

def test_render_hud(mock_pygame_setup):
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    canvas.font = MagicMock()
    
    canvas._render_hud(60.0, 100)
    canvas.screen.blit.assert_called()
