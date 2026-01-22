import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_laplacian__24_big_endian(self):
    """ """
    pygame.display.init()
    try:
        surf_1 = pygame.image.load(example_path(os.path.join('data', 'laplacian.png')))
        SIZE = 32
        surf_2 = pygame.Surface((SIZE, SIZE), 0, 24)
        pygame.transform.laplacian(surface=surf_1, dest_surface=surf_2)
        self.assertEqual(surf_2.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(surf_2.get_at((3, 10)), (255, 0, 0, 255))
        self.assertEqual(surf_2.get_at((0, 31)), (255, 0, 0, 255))
        self.assertEqual(surf_2.get_at((31, 31)), (255, 0, 0, 255))
        surf_2 = pygame.transform.laplacian(surf_1)
        self.assertEqual(surf_2.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(surf_2.get_at((3, 10)), (255, 0, 0, 255))
        self.assertEqual(surf_2.get_at((0, 31)), (255, 0, 0, 255))
        self.assertEqual(surf_2.get_at((31, 31)), (255, 0, 0, 255))
    finally:
        pygame.display.quit()