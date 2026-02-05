import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pygame
import pygame_gui
import json

from pyparticles.core.types import SimulationConfig, InteractionRule, ForceType

import importlib
import pyparticles.ui.gui_v2

@pytest.fixture
def mock_deps():
    def create_mock_element(*args, **kwargs):
        element = MagicMock()
        element.set_relative_position = MagicMock()
        element.set_text = MagicMock()
        return element

    def create_mock_slider(*args, **kwargs):
        slider = MagicMock()
        slider.set_relative_position = MagicMock()
        slider.set_current_value = MagicMock()
        slider.get_current_value.return_value = 1.0
        return slider

    with patch("pygame_gui.elements.UIPanel"), \
         patch("pygame_gui.elements.UILabel", side_effect=create_mock_element), \
         patch("pygame_gui.elements.UIHorizontalSlider", side_effect=create_mock_slider), \
         patch("pygame_gui.elements.UIButton", side_effect=create_mock_element), \
         patch("pygame_gui.elements.UIWindow"), \
         patch("pygame_gui.elements.UIDropDownMenu", side_effect=create_mock_element), \
         patch("pygame_gui.elements.UITextEntryLine"), \
         patch("pygame_gui.elements.UISelectionList"):
        # Reload gui module to pick up patched pygame_gui classes
        importlib.reload(pyparticles.ui.gui_v2)
        yield
        # Reload again to restore real classes
        importlib.reload(pyparticles.ui.gui_v2)

def test_gui_update(mock_deps):
    from pyparticles.ui.gui_v2 import SimulationGUI
    manager = MagicMock()
    cfg = SimulationConfig.default()

def test_gui_persistence(mock_deps):
    """Test Save/Load logic."""
    from pyparticles.ui.gui_v2 import SimulationGUI
    manager = MagicMock()
    cfg = SimulationConfig.default()
    physics = MagicMock()
    physics.rules = [InteractionRule("R1", ForceType.LINEAR, np.zeros((6,6)), 0.1, 0.01)]
    physics.species_config.radius = np.array([0.1]*6)
    physics.species_config.wave_freq = np.array([3.0]*6)
    physics.exclusion_enabled = True
    physics.exclusion_strength = 8.0
    physics.spin_flip_enabled = True
    physics.spin_enabled = True
    physics.spin_coupling_strength = 0.5
    physics.state.active = 100
    physics.state.vel = np.zeros((100, 2))
    
    gui = SimulationGUI(manager, cfg, physics)
    
    # 1. Save
    event_save = MagicMock()
    event_save.type = pygame_gui.UI_BUTTON_PRESSED
    event_save.ui_element = gui.save_btn
    
    with patch("builtins.open", mock_open()) as m:
        with patch("json.dump") as mock_json:
            gui.handle_event(event_save)
            mock_json.assert_called()
            args, _ = mock_json.call_args
            data = args[0]
            assert "simulation" in data
            assert "rules" in data
            assert "physics" in data

    # 2. Load
    # Mock File Dialog pick
    event_pick = MagicMock()
    event_pick.type = pygame_gui.UI_FILE_DIALOG_PATH_PICKED
    event_pick.ui_element = gui.file_dialog # Initially None
    event_pick.text = "test.json"
    
    # Needs dialog open
    gui.file_dialog = MagicMock()
    event_pick.ui_element = gui.file_dialog
    
    load_data = {
        "simulation": {"num_particles": 100, "world_size": 50.0},
        "physics": {"exclusion_enabled": False},
        "rules": []
    }
    
    with patch("builtins.open", mock_open(read_data=json.dumps(load_data))) as m:
        with patch("json.load", return_value=load_data):
            gui.handle_event(event_pick)
            
            # Check physics updated
            physics.set_active_count.assert_called()
