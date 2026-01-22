from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def get_sprites_at(self, pos):
    """return a list with all sprites at that position

        LayeredUpdates.get_sprites_at(pos): return colliding_sprites

        Bottom sprites are listed first; the top ones are listed last.

        """
    _sprites = self._spritelist
    rect = Rect(pos, (1, 1))
    colliding_idx = rect.collidelistall(_sprites)
    return [_sprites[i] for i in colliding_idx]