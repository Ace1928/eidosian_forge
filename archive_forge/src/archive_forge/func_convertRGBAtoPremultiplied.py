import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
def convertRGBAtoPremultiplied(surface_to_modify):
    for x in range(surface_to_modify.get_width()):
        for y in range(surface_to_modify.get_height()):
            color = surface_to_modify.get_at((x, y))
            premult_color = (color[0] * color[3] / 255, color[1] * color[3] / 255, color[2] * color[3] / 255, color[3])
            surface_to_modify.set_at((x, y), premult_color)