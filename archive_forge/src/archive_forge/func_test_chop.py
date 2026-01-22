import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_chop(self):
    original_surface = pygame.Surface((20, 20))
    pygame.draw.rect(original_surface, (255, 0, 0), (0, 0, 10, 10))
    pygame.draw.rect(original_surface, (0, 255, 0), (0, 10, 10, 10))
    pygame.draw.rect(original_surface, (0, 0, 255), (10, 0, 10, 10))
    pygame.draw.rect(original_surface, (255, 255, 0), (10, 10, 10, 10))
    rect = pygame.Rect(0, 0, 5, 15)
    test_surface = pygame.transform.chop(original_surface, rect)
    self.assertEqual(test_surface.get_size(), (15, 5))
    for x in range(15):
        for y in range(5):
            if x < 5:
                self.assertEqual(test_surface.get_at((x, y)), (0, 255, 0))
            else:
                self.assertEqual(test_surface.get_at((x, y)), (255, 255, 0))
    self.assertEqual(original_surface.get_size(), (20, 20))
    for x in range(20):
        for y in range(20):
            if x < 10 and y < 10:
                self.assertEqual(original_surface.get_at((x, y)), (255, 0, 0))
            if x < 10 < y:
                self.assertEqual(original_surface.get_at((x, y)), (0, 255, 0))
            if x > 10 > y:
                self.assertEqual(original_surface.get_at((x, y)), (0, 0, 255))
            if x > 10 and y > 10:
                self.assertEqual(original_surface.get_at((x, y)), (255, 255, 0))
    rect = pygame.Rect(0, 0, 10, 10)
    rect.center = original_surface.get_rect().center
    test_surface = pygame.transform.chop(surface=original_surface, rect=rect)
    self.assertEqual(test_surface.get_size(), (10, 10))
    for x in range(10):
        for y in range(10):
            if x < 5 and y < 5:
                self.assertEqual(test_surface.get_at((x, y)), (255, 0, 0))
            if x < 5 < y:
                self.assertEqual(test_surface.get_at((x, y)), (0, 255, 0))
            if x > 5 > y:
                self.assertEqual(test_surface.get_at((x, y)), (0, 0, 255))
            if x > 5 and y > 5:
                self.assertEqual(test_surface.get_at((x, y)), (255, 255, 0))
    rect = pygame.Rect(10, 10, 0, 0)
    test_surface = pygame.transform.chop(original_surface, rect)
    self.assertEqual(test_surface.get_size(), (20, 20))
    rect = pygame.Rect(0, 0, 20, 20)
    test_surface = pygame.transform.chop(original_surface, rect)
    self.assertEqual(test_surface.get_size(), (0, 0))
    rect = pygame.Rect(5, 15, 20, 20)
    test_surface = pygame.transform.chop(original_surface, rect)
    self.assertEqual(test_surface.get_size(), (5, 15))
    rect = pygame.Rect(400, 400, 10, 10)
    test_surface = pygame.transform.chop(original_surface, rect)
    self.assertEqual(test_surface.get_size(), (20, 20))