import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pygame
import pygame_gui
from pyparticles.core.types import SimulationConfig, InteractionRule, ForceType
from pyparticles.ui.gui import SimulationGUI

@pytest.fixture
def mock_deps():
    with patch("pyparticles.ui.gui.UIPanel"), \
         patch("pyparticles.ui.gui.UILabel"), \
         patch("pyparticles.ui.gui.UIHorizontalSlider"), \
         patch("pyparticles.ui.gui.UIButton"), \
         patch("pyparticles.ui.gui.UIWindow"), \
         patch("pyparticles.ui.gui.UIDropDownMenu"):
        yield

def test_gui_init(mock_deps):
    manager = MagicMock()
    cfg = SimulationConfig.default()
    physics = MagicMock()
    # Mock rules
    r1 = InteractionRule("R1", ForceType.LINEAR, np.zeros((6,6)), 0.1, 0.01)
    physics.rules = [r1]
    
    gui = SimulationGUI(manager, cfg, physics)
    assert gui.paused is False
    assert len(gui.matrix_buttons) == 36

def test_gui_update(mock_deps):
    manager = MagicMock()
    cfg = SimulationConfig.default()
    physics = MagicMock()
    r1 = InteractionRule("R1", ForceType.LINEAR, np.zeros((6,6)), 0.1, 0.01)
    physics.rules = [r1]
    
    gui = SimulationGUI(manager, cfg, physics)
    
    gui.friction_slider = MagicMock()
    gui.friction_slider.get_current_value.return_value = 0.9
    
    gui.dt_slider = MagicMock()
    gui.dt_slider.get_current_value.return_value = 0.05
    
    gui.radius_slider = MagicMock()
    gui.radius_slider.get_current_value.return_value = 0.2
    
    gui.update(0.1)
    
    assert cfg.friction == 0.9
    assert cfg.dt == 0.05
    assert physics.rules[0].max_radius == 0.2

def test_gui_events(mock_deps):
    with patch("pyparticles.ui.gui.pygame_gui") as mock_pg_gui:
        mock_pg_gui.UI_BUTTON_PRESSED = 1
        mock_pg_gui.UI_DROP_DOWN_MENU_CHANGED = 2
        
        manager = MagicMock()
        cfg = SimulationConfig.default()
        physics = MagicMock()
        # Two rules
        r1 = InteractionRule("R1", ForceType.LINEAR, np.zeros((6,6)), 0.1, 0.01)
        r2 = InteractionRule("R2", ForceType.INVERSE_SQUARE, np.ones((6,6)), 0.5, 0.01)
        physics.rules = [r1, r2]
        
        gui = SimulationGUI(manager, cfg, physics)
        
        # 1. Test Rule Switching
        event_switch = MagicMock()
        event_switch.type = 2
        event_switch.ui_element = gui.rule_selector
        event_switch.text = "R2"
        
        gui.handle_event(event_switch)
        assert gui.active_rule_idx == 1
        # Check if button updated (mocked, but logic ran)
        
        # 2. Test Matrix Button (on Active Rule R2)
        btn = gui.matrix_buttons[(0, 0)]
        event_btn = MagicMock()
        event_btn.type = 1
        event_btn.ui_element = btn
        
        # R2 matrix is ones -> 1.0. Should cycle to 0.5.
        gui.handle_event(event_btn)
        assert physics.rules[1].matrix[0, 0] == 0.5
        
        # 3. Test Save
        event_save = MagicMock()
        event_save.type = 1
        event_save.ui_element = gui.save_btn
        
        with patch("builtins.open", mock_open()) as m:
            with patch("json.dump") as mock_json:
                 gui.handle_event(event_save)
                 m.assert_called_with("rules_config.json", "w")
