from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def spritecollideany(sprite, group, collided=None):
    """finds any sprites in a group that collide with the given sprite

    pygame.sprite.spritecollideany(sprite, group): return sprite

    Given a sprite and a group of sprites, this will return return any single
    sprite that collides with with the given sprite. If there are no
    collisions, then this returns None.

    If you don't need all the features of the spritecollide function, this
    function will be a bit quicker.

    Collided is a callback function used to calculate if two sprites are
    colliding. It should take two sprites as values and return a bool value
    indicating if they are colliding. If collided is not passed, then all
    sprites must have a "rect" value, which is a rectangle of the sprite area,
    which will be used to calculate the collision.


    """
    default_sprite_collide_func = sprite.rect.colliderect
    if collided is not None:
        for group_sprite in group:
            if collided(sprite, group_sprite):
                return group_sprite
    else:
        for group_sprite in group:
            if default_sprite_collide_func(group_sprite.rect):
                return group_sprite
    return None