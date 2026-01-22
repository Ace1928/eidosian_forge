import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_get_active(self):
    """Test the get_active function"""
    pygame.display.quit()
    self.assertEqual(pygame.display.get_active(), False)
    pygame.display.init()
    pygame.display.set_mode((640, 480))
    self.assertEqual(pygame.display.get_active(), True)
    pygame.display.quit()
    pygame.display.init()
    self.assertEqual(pygame.display.get_active(), False)