import unittest
import os
import platform
import warnings
import pygame
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER', '') == 'dummy', 'Cursors not supported on headless test machines')
def test_set_cursor(self):
    """Ensures set_cursor works correctly."""
    size = (8, 8)
    hotspot = (0, 0)
    xormask = (0, 126, 64, 64, 32, 16, 0, 0)
    andmask = (254, 255, 254, 112, 56, 28, 12, 0)
    bitmap_cursor = pygame.cursors.Cursor(size, hotspot, xormask, andmask)
    constant = pygame.SYSTEM_CURSOR_ARROW
    system_cursor = pygame.cursors.Cursor(constant)
    surface = pygame.Surface((10, 10))
    color_cursor = pygame.cursors.Cursor(hotspot, surface)
    pygame.display.quit()
    with self.assertRaises(pygame.error):
        pygame.mouse.set_cursor(bitmap_cursor)
    with self.assertRaises(pygame.error):
        pygame.mouse.set_cursor(system_cursor)
    with self.assertRaises(pygame.error):
        pygame.mouse.set_cursor(color_cursor)
    pygame.display.init()
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(('w', 'h'), hotspot, xormask, andmask)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(size, ('0', '0'), xormask, andmask)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(size, ('x', 'y', 'z'), xormask, andmask)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(size, hotspot, 12345678, andmask)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(size, hotspot, xormask, 12345678)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(size, hotspot, '00000000', andmask)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(size, hotspot, xormask, (2, [0], 4, 0, 0, 8, 0, 1))
    with self.assertRaises(ValueError):
        pygame.mouse.set_cursor((3, 8), hotspot, xormask, andmask)
    with self.assertRaises(ValueError):
        pygame.mouse.set_cursor((16, 2), hotspot, (128, 64, 32), andmask)
    with self.assertRaises(ValueError):
        pygame.mouse.set_cursor((16, 2), hotspot, xormask, (192, 96, 48, 0, 1))
    self.assertEqual(pygame.mouse.set_cursor((16, 1), hotspot, (8, 0), (0, 192)), None)
    pygame.mouse.set_cursor(size, hotspot, xormask, andmask)
    self.assertEqual(pygame.mouse.get_cursor(), bitmap_cursor)
    pygame.mouse.set_cursor(size, hotspot, list(xormask), list(andmask))
    self.assertEqual(pygame.mouse.get_cursor(), bitmap_cursor)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(-50021232)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor('yellow')
    self.assertEqual(pygame.mouse.set_cursor(constant), None)
    pygame.mouse.set_cursor(constant)
    self.assertEqual(pygame.mouse.get_cursor(), system_cursor)
    pygame.mouse.set_cursor(system_cursor)
    self.assertEqual(pygame.mouse.get_cursor(), system_cursor)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(('x', 'y'), surface)
    with self.assertRaises(TypeError):
        pygame.mouse.set_cursor(hotspot, 'not_a_surface')
    self.assertEqual(pygame.mouse.set_cursor(hotspot, surface), None)
    pygame.mouse.set_cursor(hotspot, surface)
    self.assertEqual(pygame.mouse.get_cursor(), color_cursor)
    pygame.mouse.set_cursor(color_cursor)
    self.assertEqual(pygame.mouse.get_cursor(), color_cursor)
    pygame.mouse.set_cursor((0, 0), pygame.Surface((20, 20)))
    cursor = pygame.mouse.get_cursor()
    self.assertEqual(cursor.type, 'color')
    self.assertEqual(cursor.data[0], (0, 0))
    self.assertEqual(cursor.data[1].get_size(), (20, 20))