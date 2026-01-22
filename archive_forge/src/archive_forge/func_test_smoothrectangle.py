import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_smoothrectangle(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Color, SmoothRectangle
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        rect = SmoothRectangle(pos=(100, 100), size=(150, 150))
    r(wid)
    filtered_points = self._filtered_points(rect.points)
    assert rect.antialiasing_line_points == filtered_points + filtered_points[:2]
    rect.size = (150, -2)
    r(wid)
    assert rect.antialiasing_line_points == []
    rect.size = (150, 2)
    r(wid)
    assert rect.antialiasing_line_points == []
    rect.size = (150, 150)
    r(wid)
    assert rect.antialiasing_line_points == filtered_points + filtered_points[:2]
    rect.texture = self._get_texture()
    r(wid)
    assert rect.antialiasing_line_points == []
    rect.source = ''
    r(wid)
    assert rect.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        rect = SmoothRectangle(pos=(100, 100), size=(150, -3))
    r(wid)
    assert rect.antialiasing_line_points == []
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        rect = SmoothRectangle(pos=(100, 100), size=(3.99, 3.99))
    r(wid)
    assert rect.antialiasing_line_points == []