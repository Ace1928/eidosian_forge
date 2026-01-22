import sys
import unittest
import platform
import pygame
def test_PgObject_GetBuffer_exception(self):
    from pygame.bufferproxy import BufferProxy
    bp = BufferProxy(1)
    self.assertRaises(ValueError, getattr, bp, 'length')