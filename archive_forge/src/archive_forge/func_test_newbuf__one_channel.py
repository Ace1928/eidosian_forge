import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
@unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
@unittest.skipIf(IS_PYPY, 'pypy no likey')
def test_newbuf__one_channel(self):
    mixer.init(22050, -16, 1)
    self._NEWBUF_export_check()