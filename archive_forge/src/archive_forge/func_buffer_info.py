import a new buffer interface.
import pygame
import pygame.newbuffer
from pygame.newbuffer import (
import unittest
import ctypes
import operator
from functools import reduce
def buffer_info(self):
    return (ctypes.addressof(self.buffer), self.shape[0])