import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_enlarged_line(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Line, Color, PushMatrix, PopMatrix, Scale, Translate
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        Line(points=(10, 10, 10, 90), width=1)
        Line(points=(20, 10, 20, 90), width=3)
        PushMatrix()
        Translate(30, 10, 1)
        Scale(3, 1, 1)
        Line(points=(0, 0, 0, 80), width=1, force_custom_drawing_method=True)
        PopMatrix()
    r(wid)