import sys
import unittest
import platform
import pygame
def test_unicode_error(self):
    pygame.set_error('你好')
    self.assertEqual('你好', pygame.get_error())