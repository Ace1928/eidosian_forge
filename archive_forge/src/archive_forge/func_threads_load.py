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
def threads_load(self, images):
    import pygame.threads
    for i in range(10):
        surfs = pygame.threads.tmap(pygame.image.load, images)
        for s in surfs:
            self.assertIsInstance(s, pygame.Surface)