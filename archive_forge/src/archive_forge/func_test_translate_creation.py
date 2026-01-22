import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_translate_creation(self):
    from kivy.graphics import Translate
    self.check_transform_works(Translate)