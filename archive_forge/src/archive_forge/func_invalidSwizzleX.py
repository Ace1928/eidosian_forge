import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def invalidSwizzleX():
    Vector3().xx = (1, 2)