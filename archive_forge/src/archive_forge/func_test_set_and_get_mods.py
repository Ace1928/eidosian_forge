import os
import time
import unittest
import pygame
import pygame.key
def test_set_and_get_mods(self):
    pygame.key.set_mods(pygame.KMOD_CTRL)
    self.assertEqual(pygame.key.get_mods(), pygame.KMOD_CTRL)
    pygame.key.set_mods(pygame.KMOD_ALT)
    self.assertEqual(pygame.key.get_mods(), pygame.KMOD_ALT)
    pygame.key.set_mods(pygame.KMOD_CTRL | pygame.KMOD_ALT)
    self.assertEqual(pygame.key.get_mods(), pygame.KMOD_CTRL | pygame.KMOD_ALT)