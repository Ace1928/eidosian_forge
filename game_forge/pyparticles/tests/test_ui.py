import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pygame
# import pygame_gui # Don't import real one, patch it

from pyparticles.core.types import SimulationConfig
from pyparticles.ui.gui import SimulationGUI

@pytest.fixture
def mock_deps():
    with patch("pyparticles.ui.gui.UIPanel"), \
         patch("pyparticles.ui.gui.UILabel"), \
         patch("pyparticles.ui.gui.UIHorizontalSlider"), \
         patch("pyparticles.ui.gui.UIButton") as MockButton, \
         patch("pyparticles.ui.gui.UIWindow"):
        yield MockButton

def test_gui_events(mock_deps):
    # Configure MockButton to return distinct mocks each time
    # We expect creation of: pause_btn, reset_btn, and 36 matrix buttons (6x6) + 1 save btn
    # Total ~40 buttons.
    # using side_effect with iterator
    
    mock_deps.side_effect = lambda *args, **kwargs: MagicMock()
    
    with patch("pyparticles.ui.gui.pygame_gui") as mock_pg_gui:
        mock_pg_gui.UI_BUTTON_PRESSED = 12345
        
        manager = MagicMock()
        cfg = SimulationConfig.default()
        physics = MagicMock()
        physics.matrix = np.zeros((6, 6))
        
        gui = SimulationGUI(manager, cfg, physics)
        
        # Test Pause
        event_pause = MagicMock()
        event_pause.type = 12345
        event_pause.ui_element = gui.pause_btn
        
        gui.handle_event(event_pause)
        assert gui.paused is True
        
        # Test Reset
        event_reset = MagicMock() # New event object
        event_reset.type = 12345
        event_reset.ui_element = gui.reset_btn
        
        gui.handle_event(event_reset)
        physics.reset.assert_called()
        
        # Test Matrix Button
        # The dictionary keys are (r, c)
        btn = gui.matrix_buttons[(0, 0)]
        event_matrix = MagicMock()
        event_matrix.type = 12345
        event_matrix.ui_element = btn
        
        gui.handle_event(event_matrix)
        assert physics.matrix[0, 0] == -0.5
        
        # Test Save
        event_save = MagicMock()
        event_save.type = 12345
        event_save.ui_element = gui.save_btn
        
        with patch("builtins.open", mock_open()) as m:
            with patch("json.dump") as mock_json:
                 gui.handle_event(event_save)
                 m.assert_called_with("matrix_config.json", "w")

def test_gui_init(mock_deps):
    manager = MagicMock()
    cfg = SimulationConfig.default()
    physics = MagicMock()
    physics.matrix = np.zeros((6, 6))
    gui = SimulationGUI(manager, cfg, physics)
    assert gui.paused is False

def test_gui_update(mock_deps):
    manager = MagicMock()
    cfg = SimulationConfig.default()
    physics = MagicMock()
    physics.matrix = np.zeros((6, 6))
    gui = SimulationGUI(manager, cfg, physics)
    
    gui.friction_slider = MagicMock()
    gui.friction_slider.get_current_value.return_value = 0.9
    
    gui.dt_slider = MagicMock()
    gui.dt_slider.get_current_value.return_value = 0.05
    
    gui.update(0.1)
    assert cfg.friction == 0.9
    assert cfg.dt == 0.05
