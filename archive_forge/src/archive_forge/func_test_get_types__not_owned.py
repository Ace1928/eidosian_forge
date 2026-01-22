import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_get_types__not_owned(self):
    """Ensures get_types works when the clipboard is not owned
        by the pygame application.
        """
    self._skip_if_clipboard_owned()
    data_types = scrap.get_types()
    self.assertIsInstance(data_types, list)