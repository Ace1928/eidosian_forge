import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def invalidSwizzleZ():
    Vector3().zz = (1, 2)