import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def check_transform_works(self, transform_type):
    transform = transform_type(0, 1, 2)
    self.assertEqual(transform.x, 0)
    self.assertEqual(transform.y, 1)
    self.assertEqual(transform.z, 2)
    transform = transform_type(x=0, y=1)
    self.assertEqual(transform.x, 0)
    self.assertEqual(transform.y, 1)
    transform = transform_type(x=0, y=1, z=2)
    self.assertEqual(transform.x, 0)
    self.assertEqual(transform.y, 1)
    self.assertEqual(transform.z, 2)