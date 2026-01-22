import unittest
from pygame.tests.test_utils import fixture_path
import pygame
def test_load_xbm(self):
    cursorfile = fixture_path('xbm_cursors/white_sizing.xbm')
    maskfile = fixture_path('xbm_cursors/white_sizing_mask.xbm')
    cursor = pygame.cursors.load_xbm(cursorfile, maskfile)
    with open(cursorfile) as cursor_f, open(maskfile) as mask_f:
        cursor = pygame.cursors.load_xbm(cursor_f, mask_f)
    import pathlib
    cursor = pygame.cursors.load_xbm(pathlib.Path(cursorfile), pathlib.Path(maskfile))
    pygame.display.init()
    try:
        pygame.mouse.set_cursor(*cursor)
    except pygame.error as e:
        if 'not currently supported' in str(e):
            unittest.skip('skipping test as set_cursor() is not supported')
    finally:
        pygame.display.quit()