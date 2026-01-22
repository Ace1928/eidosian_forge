import unittest
import pygame
from pygame import sprite
def test_nondirty_intersections_redrawn__with_source_rect(self):
    """Ensure non-dirty sprites using source_rects are correctly redrawn
        when dirty sprites intersect with them.

        Related to issue #898.
        """
    self._nondirty_intersections_redrawn(True)