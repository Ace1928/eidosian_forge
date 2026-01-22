import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_list_modes(self):
    modes = pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN, display=0)
    if modes != -1:
        self.assertEqual(len(modes[0]), 2)
        self.assertEqual(type(modes[0][0]), int)
    modes = pygame.display.list_modes()
    if modes != -1:
        self.assertEqual(len(modes[0]), 2)
        self.assertEqual(type(modes[0][0]), int)
        self.assertEqual(len(modes), len(set(modes)))
    modes = pygame.display.list_modes(depth=0, flags=0, display=0)
    if modes != -1:
        self.assertEqual(len(modes[0]), 2)
        self.assertEqual(type(modes[0][0]), int)