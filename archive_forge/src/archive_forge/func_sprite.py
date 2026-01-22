from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
@sprite.setter
def sprite(self, sprite_to_set):
    self._set_sprite(sprite_to_set)