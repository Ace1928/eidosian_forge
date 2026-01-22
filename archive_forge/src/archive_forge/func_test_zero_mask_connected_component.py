from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_connected_component(self):
    """Ensures connected_component correctly handles zero sized masks."""
    expected_count = 0
    for size in ((81, 0), (0, 80), (0, 0)):
        msg = f'size={size}'
        mask = pygame.mask.Mask(size)
        cc_mask = mask.connected_component()
        self.assertIsInstance(cc_mask, pygame.mask.Mask, msg)
        self.assertEqual(cc_mask.get_size(), size)
        self.assertEqual(cc_mask.count(), expected_count, msg)