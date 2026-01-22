import unittest
import pygame
from pygame import sprite
def test_repaint_rect(self):
    group = self.LG
    surface = pygame.Surface((100, 100))
    group.repaint_rect(pygame.Rect(0, 0, 100, 100))
    group.draw(surface)