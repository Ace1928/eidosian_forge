import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') in skip_list, 'requires the SDL_VIDEODRIVER to be non dummy')
def test_toggle_fullscreen(self):
    """Test for toggle fullscreen"""
    pygame.display.quit()
    with self.assertRaises(pygame.error):
        pygame.display.toggle_fullscreen()
    pygame.display.init()
    width_height = (640, 480)
    test_surf = pygame.display.set_mode(width_height)
    try:
        pygame.display.toggle_fullscreen()
    except pygame.error:
        self.fail()
    else:
        if pygame.display.toggle_fullscreen() == 1:
            boolean = (test_surf.get_width(), test_surf.get_height()) in pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN, display=0)
            self.assertEqual(boolean, True)
        else:
            self.assertEqual((test_surf.get_width(), test_surf.get_height()), width_height)