import unittest
import os
import platform
import warnings
import pygame
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER', '') == 'dummy', 'mouse.set_system_cursor only available in SDL2')
def test_set_system_cursor(self):
    """Ensures set_system_cursor works correctly."""
    with warnings.catch_warnings(record=True) as w:
        'From Pygame 2.0.1, set_system_cursor() should raise a deprecation warning'
        warnings.simplefilter('always')
        with self.assertRaises(pygame.error):
            pygame.display.quit()
            pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_HAND)
        pygame.display.init()
        with self.assertRaises(TypeError):
            pygame.mouse.set_system_cursor('b')
        with self.assertRaises(TypeError):
            pygame.mouse.set_system_cursor(None)
        with self.assertRaises(TypeError):
            pygame.mouse.set_system_cursor((8, 8), (0, 0))
        with self.assertRaises(pygame.error):
            pygame.mouse.set_system_cursor(2000)
        self.assertEqual(pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_ARROW), None)
        self.assertEqual(len(w), 6)
        self.assertTrue(all([issubclass(warn.category, DeprecationWarning) for warn in w]))