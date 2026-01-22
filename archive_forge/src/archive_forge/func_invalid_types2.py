import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def invalid_types2():
    origin = Vector2(7.22, 2004.0)
    target = Vector2(12.3, 2021.0)
    origin.move_towards('kinda', 3)