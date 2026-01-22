from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_to_surface__create_surface(self):
    """Ensures to_surface correctly handles zero sized masks and surfaces
        when it has to create a default surface.
        """
    mask_color = pygame.Color('blue')
    for mask_size in ((3, 0), (0, 3), (0, 0)):
        mask = pygame.mask.Mask(mask_size, fill=True)
        to_surface = mask.to_surface(setcolor=mask_color)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), mask_size)