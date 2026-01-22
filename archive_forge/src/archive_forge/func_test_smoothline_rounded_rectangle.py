import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_smoothline_rounded_rectangle(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import SmoothLine, Color
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        line = SmoothLine(rounded_rectangle=(100, 100, 0.5, 1.99, 30, 30, 30, 30, 100))
    r(wid)
    assert line.rounded_rectangle is None