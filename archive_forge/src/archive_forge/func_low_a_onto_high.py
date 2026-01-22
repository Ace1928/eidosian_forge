import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def low_a_onto_high(high, low):
    """Tests straight alpha case. Source is high alpha, destination is low alpha"""
    high_alpha_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
    low_alpha_surface = high_alpha_surface.copy()
    high_alpha_color = Color((high, high, low, high))
    low_alpha_color = Color((high, low, low, low))
    high_alpha_surface.fill(high_alpha_color)
    low_alpha_surface.fill(low_alpha_color)
    low_alpha_surface.blit(high_alpha_surface, (0, 0))
    expected_color = high_alpha_color + Color(tuple((x * (255 - high_alpha_color.a) // 255 for x in low_alpha_color)))
    self.assertTrue(check_color_diff(low_alpha_surface.get_at((0, 0)), expected_color))