import os
import pygame as pg
def punch(self, target):
    """returns true if the fist collides with the target"""
    if not self.punching:
        self.punching = True
        hitbox = self.rect.inflate(-5, -5)
        return hitbox.colliderect(target.rect)