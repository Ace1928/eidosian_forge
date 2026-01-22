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
def test_get_sdl_image_version(self):
    if not pygame.image.get_extended():
        self.assertIsNone(pygame.image.get_sdl_image_version())
    else:
        expected_length = 3
        expected_type = tuple
        expected_item_type = int
        version = pygame.image.get_sdl_image_version()
        self.assertIsInstance(version, expected_type)
        self.assertEqual(len(version), expected_length)
        for item in version:
            self.assertIsInstance(item, expected_item_type)