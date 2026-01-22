import sys
import unittest
import platform
import pygame
def not_init_assertions(self):
    self.assertFalse(pygame.get_init(), "pygame shouldn't be initialized")
    self.assertFalse(pygame.display.get_init(), "display shouldn't be initialized")
    if 'pygame.mixer' in sys.modules:
        self.assertFalse(pygame.mixer.get_init(), "mixer shouldn't be initialized")
    if 'pygame.font' in sys.modules:
        self.assertFalse(pygame.font.get_init(), "init shouldn't be initialized")
    import platform
    if platform.system().startswith('Darwin'):
        return
    try:
        self.assertRaises(pygame.error, pygame.scrap.get)
    except NotImplementedError:
        pass