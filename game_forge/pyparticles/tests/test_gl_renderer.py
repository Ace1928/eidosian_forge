"""
Tests for OpenGL Renderer (Mocked).
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pyparticles.core.types import SimulationConfig

@pytest.fixture
def mock_moderngl():
    with patch("pyparticles.rendering.gl_renderer.moderngl") as mock_gl:
        mock_ctx = MagicMock()
        mock_gl.create_context.return_value = mock_ctx
        yield mock_gl

def test_gl_renderer_init(mock_moderngl):
    from pyparticles.rendering.gl_renderer import GLCanvas
    cfg = SimulationConfig.default()
    
    # Mock open for shaders
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "void main() {}"
        renderer = GLCanvas(cfg)
        
    assert renderer.ctx.create_context.called is False # moderngl.create_context called directly
    assert renderer.prog is not None
    assert renderer.vbo is not None

def test_gl_renderer_render(mock_moderngl):
    from pyparticles.rendering.gl_renderer import GLCanvas
    cfg = SimulationConfig.default()
    
    with patch("builtins.open") as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "src"
        renderer = GLCanvas(cfg)
        
    # Mock VBO write
    renderer.vbo = MagicMock()
    renderer.vao = MagicMock()
    
    pos = np.zeros((100, 2), dtype=np.float32)
    vel = np.zeros((100, 2), dtype=np.float32)
    colors = np.zeros(100, dtype=np.int32)
    angle = np.zeros(100, dtype=np.float32)
    species = np.zeros((6, 3), dtype=np.float32)
    
    renderer.render(pos, vel, colors, angle, species, 50, 60.0)
    
    renderer.vbo.write.assert_called()
    renderer.vao.render.assert_called()
