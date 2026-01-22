import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_screensaver_support(self):
    pygame.display.set_allow_screensaver(True)
    self.assertTrue(pygame.display.get_allow_screensaver())
    pygame.display.set_allow_screensaver(False)
    self.assertFalse(pygame.display.get_allow_screensaver())
    pygame.display.set_allow_screensaver()
    self.assertTrue(pygame.display.get_allow_screensaver())