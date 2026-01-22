import re
import weakref
import gc
import ctypes
import unittest
import pygame
from pygame.bufferproxy import BufferProxy
def test_parent_property(self):
    kwds = dict(self.view_keywords)
    p = []
    kwds['parent'] = p
    v = BufferProxy(kwds)
    self.assertIs(v.parent, p)