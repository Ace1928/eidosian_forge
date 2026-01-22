import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_get_smoothscale_backend(self):
    filter_type = pygame.transform.get_smoothscale_backend()
    self.assertTrue(filter_type in ['GENERIC', 'MMX', 'SSE'])