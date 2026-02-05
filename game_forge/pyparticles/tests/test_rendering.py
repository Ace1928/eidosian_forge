import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pyparticles.core.types import SimulationConfig, RenderMode
from pyparticles.rendering.canvas import Canvas

@pytest.fixture
def mock_pygame_setup():
    with patch("pygame.init"), \
         patch("pygame.display.set_mode"), \
         patch("pygame.display.set_caption"), \
         patch("pygame.font.SysFont"):
        yield

def test_render_sprites(mock_pygame_setup):
    """Test standard sprite rendering path."""
    cfg = SimulationConfig.default()
    cfg.render_mode = RenderMode.SPRITES
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    
    pos = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    colors = np.array([0, 1], dtype=np.int32)
    active = 2
    
    canvas.render(pos, colors, active, 60.0)
    
    # Check blits called
    canvas.screen.blits.assert_called()
    # Arg should be a list of (surf, (x, y))
    call_arg = canvas.screen.blits.call_args[0][0]
    assert len(call_arg) == 2

def test_render_glow(mock_pygame_setup):
    """Test glow mode initialization and render."""
    cfg = SimulationConfig.default()
    cfg.render_mode = RenderMode.GLOW
    canvas = Canvas(cfg)
    
    # Check sprite size (glow is larger)
    # radius_px is calculated based on screen width. 1200 -> scale 600. max_r 0.1 * 0.3 * 600 = 18.
    # Glow sprite size = r * 4 = 72
    assert canvas.sprites[0].get_width() > canvas.radius_px * 2
    
    canvas.screen = MagicMock()
    pos = np.array([[10.0, 10.0]], dtype=np.float32)
    colors = np.array([0], dtype=np.int32)
    canvas.render(pos, colors, 1, 60.0)
    canvas.screen.blits.assert_called()

def test_render_points_logic(mock_pygame_setup):
    """Test _render_fast_points logic with mocked PixelArray."""
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    canvas.screen.map_rgb.return_value = 16777215
    
    mock_px_array = MagicMock()
    with patch("pygame.PixelArray", return_value=mock_px_array) as MockPixelArray:
        sx = np.array([10.0, 20.0, -5.0], dtype=np.float32)
        sy = np.array([10.0, 20.0, 10.0], dtype=np.float32)
        colors = np.array([0, 1, 0], dtype=np.int32)
        
        canvas._render_fast_points(sx, sy, colors)
        
        MockPixelArray.assert_called_with(canvas.screen)
        mock_px_array.close.assert_called()
        assert mock_px_array.__setitem__.call_count >= 2

def test_render_points_exception(mock_pygame_setup):
    """Test exception handling in fast points."""
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    
    with patch("pygame.PixelArray", side_effect=ValueError("Locked")):
        # Should catch exception and not crash
        canvas._render_fast_points(np.zeros(1), np.zeros(1), np.zeros(1, dtype=int))

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
    
    canvas.render(pos, colors, 10001, 60.0)
    
    canvas._render_fast_points.assert_called()
    canvas._render_sprites.assert_not_called()

def test_resize_logic(mock_pygame_setup):
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.resize(100, 100)
    assert canvas.scale == 50.0
    assert canvas.offset_x == 50.0

def test_render_hud(mock_pygame_setup):
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    canvas.font = MagicMock()
    
    canvas._render_hud(60.0, 100)
    canvas.screen.blit.assert_called()