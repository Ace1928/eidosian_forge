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
def test_load_extended(self):
    """can load different format images.

        We test loading the following file types:
            bmp, png, jpg, gif (non-animated), pcx, tga (uncompressed), tif, xpm, ppm, pgm
        Following file types are tested when using SDL 2
            svg, pnm, webp
        All the loaded images are smaller than 32 x 32 pixels.
        """
    filename_expected_color = [('asprite.bmp', (255, 255, 255, 255)), ('laplacian.png', (10, 10, 70, 255)), ('red.jpg', (254, 0, 0, 255)), ('blue.gif', (0, 0, 255, 255)), ('green.pcx', (0, 255, 0, 255)), ('yellow.tga', (255, 255, 0, 255)), ('turquoise.tif', (0, 255, 255, 255)), ('purple.xpm', (255, 0, 255, 255)), ('black.ppm', (0, 0, 0, 255)), ('grey.pgm', (120, 120, 120, 255)), ('teal.svg', (0, 128, 128, 255)), ('crimson.pnm', (220, 20, 60, 255)), ('scarlet.webp', (252, 14, 53, 255))]
    for filename, expected_color in filename_expected_color:
        if filename.endswith('svg') and sdl_image_svg_jpeg_save_bug:
            continue
        with self.subTest(f'Test loading a {filename.split('.')[-1]}', filename='examples/data/' + filename, expected_color=expected_color):
            surf = pygame.image.load_extended(example_path('data/' + filename))
            self.assertEqual(surf.get_at((0, 0)), expected_color)