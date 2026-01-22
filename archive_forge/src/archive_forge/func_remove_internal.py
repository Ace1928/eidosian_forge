from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def remove_internal(self, sprite):
    if sprite is self.__sprite:
        self.__sprite = None
    if sprite in self.spritedict:
        AbstractGroup.remove_internal(self, sprite)