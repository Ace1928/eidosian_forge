import pytest
from unittest.mock import patch, MagicMock
import sys
import pygame
from pyparticles.app import main

def test_app_main():
    test_args = ["pyparticles", "--num", "10", "--mode", "sprites", "--jit-warmup"]
    with patch.object(sys, 'argv', test_args):
        with patch("pyparticles.app.PhysicsEngine"), \
             patch("pyparticles.app.Canvas") as MockCanvas, \
             patch("pyparticles.app.pygame_gui.UIManager") as MockManager, \
             patch("pyparticles.app.SimulationGUI") as MockGUI, \
             patch("pygame.time.Clock"), \
             patch("pygame.event.get"), \
             patch("pygame.display.flip"), \
             patch("pygame.quit"):
            
            # Setup Events to cover resize, keydown, pause
            
            # Event 1: Resize
            evt_resize = MagicMock()
            evt_resize.type = 32777 # VIDEORESIZE (usually, or we patch pygame constants)
            evt_resize.w = 800
            evt_resize.h = 600
            
            # Event 2: Keydown Space (Pause)
            evt_space = MagicMock()
            evt_space.type = 768 # KEYDOWN
            evt_space.key = 32 # SPACE
            
            # Event 3: Keydown Escape (Quit)
            evt_esc = MagicMock()
            evt_esc.type = 768
            evt_esc.key = 27 # ESC
            
            # Event 4: Quit
            evt_quit = MagicMock()
            evt_quit.type = 256 # QUIT
            
            # We need to supply constants because 'pygame.VIDEORESIZE' inside app.py needs to match
            # Since we import real pygame in test, we can use real constants.
            evt_resize.type = pygame.VIDEORESIZE
            evt_space.type = pygame.KEYDOWN
            evt_space.key = pygame.K_SPACE
            evt_esc.type = pygame.KEYDOWN
            evt_esc.key = pygame.K_ESCAPE
            evt_quit.type = pygame.QUIT
            
            # Sequence of event lists returned by pygame.event.get()
            side_effect = [
                [evt_resize], 
                [evt_space],
                [evt_space], # Toggle pause back
                [evt_quit] # End loop
            ]
            
            with patch("pygame.event.get", side_effect=side_effect):
                main()
            
            # Assertions
            MockCanvas.return_value.resize.assert_called_with(800, 600)
            MockGUI.return_value.pause_btn.set_text.assert_called()