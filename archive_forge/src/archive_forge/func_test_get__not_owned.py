import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_get__not_owned(self):
    """Ensures get works when there is no data of the requested type
        in the clipboard and the clipboard is not owned by the pygame
        application.
        """
    self._skip_if_clipboard_owned()
    DATA_TYPE = 'test_get__not_owned'
    data = scrap.get(DATA_TYPE)
    self.assertIsNone(data)