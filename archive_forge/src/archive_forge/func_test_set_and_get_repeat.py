import os
import time
import unittest
import pygame
import pygame.key
def test_set_and_get_repeat(self):
    self.assertEqual(pygame.key.get_repeat(), (0, 0))
    pygame.key.set_repeat(10, 15)
    self.assertEqual(pygame.key.get_repeat(), (10, 15))
    pygame.key.set_repeat()
    self.assertEqual(pygame.key.get_repeat(), (0, 0))