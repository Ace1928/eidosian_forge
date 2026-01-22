import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_ellipse(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Ellipse, Color
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        self.e = Ellipse(pos=(100, 100), size=(200, 100))
    r(wid)