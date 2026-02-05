import pytest
from unittest.mock import patch, MagicMock
import sys
import pygame
from pyparticles.app import main

def test_app_main():
    test_args = ["pyparticles", "--num", "10", "--jit-warmup"]
    with patch.object(sys, 'argv', test_args):
        with patch("pyparticles.app.PhysicsEngine"), \
             patch("pyparticles.app.pygame_gui.UIManager") as MockManager, \
             patch("pyparticles.app.SimulationGUI") as MockGUI, \
             patch("pyparticles.app.GLCanvas") as MockGLCanvas, \
             patch("pygame.time.Clock") as MockClock, \
             patch("pygame.event.get") as MockEventGet, \
             patch("pygame.display.flip"), \
             patch("pygame.quit"), \
             patch("pygame.init"), \
             patch("pygame.display.set_mode"), \
             patch("pygame.display.gl_set_attribute"):
            
            # Configure Clock
            MockClock.return_value.tick.return_value = 16 # 16ms
            MockClock.return_value.get_fps.return_value = 60.0
            
            # Setup Events
            evt_resize = MagicMock()
            evt_resize.type = pygame.VIDEORESIZE
            evt_resize.w = 800
            evt_resize.h = 600
            
            evt_space = MagicMock()
            evt_space.type = pygame.KEYDOWN
            evt_space.key = pygame.K_SPACE
            
            evt_quit = MagicMock()
            evt_quit.type = pygame.QUIT
            
            MockEventGet.side_effect = [
                [evt_resize], 
                [evt_space],
                [evt_space],
                [evt_quit]
            ]
            
            # Mock GUI properties needed by app loop
            MockGUI.return_value.pause_btn = MagicMock()
            MockGUI.return_value.fps_label = MagicMock()
            
            main()
            
            # Assertions
            MockGLCanvas.return_value.resize.assert_called_with(800, 600)
            MockGUI.return_value.pause_btn.set_text.assert_called()
