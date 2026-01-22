import unittest
import platform
@unittest.skipIf('Darwin' in platform.platform() or 'Windows' in platform.platform(), 'Not unix we skip.')
def test_initsysfonts_unix(self):
    import pygame.sysfont
    self.assertTrue(len(pygame.sysfont.get_fonts()) > 0)