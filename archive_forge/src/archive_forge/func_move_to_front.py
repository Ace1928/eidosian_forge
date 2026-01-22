from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def move_to_front(self, sprite):
    """bring the sprite to front layer

        LayeredUpdates.move_to_front(sprite): return None

        Brings the sprite to front by changing the sprite layer to the top-most
        layer. The sprite is added at the end of the list of sprites in that
        top-most layer.

        """
    self.change_layer(sprite, self.get_top_layer())