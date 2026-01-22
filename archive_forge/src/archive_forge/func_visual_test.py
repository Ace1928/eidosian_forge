import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def visual_test(self, fullscreen=False):
    text = ''
    if fullscreen:
        if not self.isfullscreen:
            pygame.display.toggle_fullscreen()
            self.isfullscreen = True
        text = 'Is this in fullscreen? [y/n]'
    else:
        if self.isfullscreen:
            pygame.display.toggle_fullscreen()
            self.isfullscreen = False
        text = 'Is this not in fullscreen [y/n]'
    s = self.font.render(text, False, (0, 0, 0))
    self.screen.blit(s, (self.WIDTH / 2 - self.font.size(text)[0] / 2, 100))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_y:
                    return True
                if event.key == pygame.K_n:
                    return False
            if event.type == pygame.QUIT:
                return False