import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pygame
import pygame_gui
import json

from pyparticles.core.types import SimulationConfig, InteractionRule, ForceType
from pyparticles.ui.gui import SimulationGUI

import importlib
import pyparticles.ui.gui

@pytest.fixture
def mock_deps():
    with patch("pygame_gui.elements.UIPanel"), \
         patch("pygame_gui.elements.UILabel"), \
         patch("pygame_gui.elements.UIHorizontalSlider"), \
         patch("pygame_gui.elements.UIButton", side_effect=MagicMock) as mock_btn, \
         patch("pygame_gui.elements.UIWindow"), \
         patch("pygame_gui.elements.UIDropDownMenu"), \
         patch("pygame_gui.elements.UITextEntryLine"), \
         patch("pygame_gui.elements.UISelectionList"):
        # Reload gui module to pick up patched pygame_gui classes
        importlib.reload(pyparticles.ui.gui)
        yield
        # Reload again to restore real classes (optional but good practice)
        importlib.reload(pyparticles.ui.gui)

def test_gui_update(mock_deps):
    from pyparticles.ui.gui import SimulationGUI
    manager = MagicMock()
    cfg = SimulationConfig.default()

def test_gui_persistence(mock_deps):
    """Test Save/Load logic."""
    from pyparticles.ui.gui import SimulationGUI
    manager = MagicMock()
    cfg = SimulationConfig.default()
    physics = MagicMock()
    physics.rules = [InteractionRule("R1", ForceType.LINEAR, np.zeros((6,6)), 0.1, 0.01)]
    physics.species_config.radius = np.array([0.1]*6)
    physics.species_config.wave_freq = np.array([3.0]*6)
    
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
            assert "species" in data
            assert data["species"]["wave_freq"][0] == 3.0

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
        "simulation": {"num_particles": 100},
        "rules": [],
        "species": {"wave_freq": [5.0]*6}
    }
    
    with patch("builtins.open", mock_open(read_data=json.dumps(load_data))) as m:
        with patch("json.load", return_value=load_data):
            gui.handle_event(event_pick)
            
            # Check physics updated
            physics.set_active_count.assert_called_with(100)
            assert np.all(physics.species_config.wave_freq == 5.0)

