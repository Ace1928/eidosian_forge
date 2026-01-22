import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_sdl_mixer_version__linked_equals_compiled(self):
    """Ensures get_sdl_mixer_version's linked/compiled versions are equal."""
    linked_version = pygame.mixer.get_sdl_mixer_version(linked=True)
    complied_version = pygame.mixer.get_sdl_mixer_version(linked=False)
    self.assertTupleEqual(linked_version, complied_version)