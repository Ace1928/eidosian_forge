import unittest
import platform
@unittest.skipIf('Darwin' not in platform.platform(), 'Not mac we skip.')
def test_initsysfonts_darwin(self):
    import pygame.sysfont
    self.assertTrue(len(pygame.sysfont.get_fonts()) > 10)