import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pygame
from pyparticles.core.types import SimulationConfig, RenderMode
from pyparticles.rendering.canvas import Canvas

@pytest.fixture
def mock_pygame_setup():
    with patch("pygame.init"), \
         patch("pygame.display.set_mode"), \
         patch("pygame.display.set_caption"), \
         patch("pygame.font.SysFont"):
        yield

def test_canvas_modes(mock_pygame_setup):
    cfg = SimulationConfig.default()
    
    # Test Sprites
    cfg.render_mode = RenderMode.SPRITES
    canvas = Canvas(cfg)
    assert len(canvas.sprites) == cfg.num_types
    
    # Test Glow
    cfg.render_mode = RenderMode.GLOW
    canvas = Canvas(cfg)
    assert len(canvas.sprites) == cfg.num_types
    
    # Test Pixels (no sprites needed technically, but init runs)
    cfg.render_mode = RenderMode.PIXELS
    canvas = Canvas(cfg)

def test_render_call(mock_pygame_setup):
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    
    pos = np.zeros((10, 2), dtype=np.float32)
    colors = np.zeros(10, dtype=np.int32)
    
    # Sprites
    cfg.render_mode = RenderMode.SPRITES
    canvas.render(pos, colors, 10, 60.0)
    canvas.screen.blits.assert_called()

def test_render_pixels_fallback(mock_pygame_setup):
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.screen = MagicMock()
    
    pos = np.zeros((10, 2), dtype=np.float32)
    colors = np.zeros(10, dtype=np.int32)
    
    # Trigger fallback by count > 10000
    # We force call _render_fast_points
    with patch("pygame.PixelArray") as MockPixelArray:
        canvas._render_fast_points(np.zeros(10), np.zeros(10), colors)
        MockPixelArray.assert_called()

def test_resize(mock_pygame_setup):
    cfg = SimulationConfig.default()
    canvas = Canvas(cfg)
    canvas.resize(800, 600)
    assert canvas.width == 800
    assert canvas.height == 600
