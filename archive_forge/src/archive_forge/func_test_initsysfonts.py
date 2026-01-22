import unittest
import platform
def test_initsysfonts(self):
    import pygame.sysfont
    pygame.sysfont.initsysfonts()
    self.assertTrue(len(pygame.sysfont.get_fonts()) > 0)