from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def get_sprite(self, idx):
    """return the sprite at the index idx from the groups sprites

        LayeredUpdates.get_sprite(idx): return sprite

        Raises IndexOutOfBounds if the idx is not within range.

        """
    return self._spritelist[idx]