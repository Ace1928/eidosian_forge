from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def remove_sprites_of_layer(self, layer_nr):
    """remove all sprites from a layer and return them as a list

        LayeredUpdates.remove_sprites_of_layer(layer_nr): return sprites

        """
    sprites = self.get_sprites_from_layer(layer_nr)
    self.remove(*sprites)
    return sprites