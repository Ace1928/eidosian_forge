import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_get_wm_info(self):
    wm_info = display.get_wm_info()
    self.assertIsInstance(wm_info, dict)
    wm_info_potential_keys = {'colorbuffer', 'connection', 'data', 'dfb', 'display', 'framebuffer', 'fswindow', 'hdc', 'hglrc', 'hinstance', 'lock_func', 'resolveFramebuffer', 'shell_surface', 'surface', 'taskHandle', 'unlock_func', 'wimpVersion', 'window', 'wmwindow'}
    wm_info_remaining_keys = set(wm_info.keys()).difference(wm_info_potential_keys)
    self.assertFalse(wm_info_remaining_keys)