import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_smoothtriangle(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Color, SmoothTriangle
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 0, 0, 0.5)
        triangle = SmoothTriangle(points=[100, 100, 200, 100, 150, 200, 500, 500, 400, 400])
    r(wid)
    filtered_points = self._filtered_points(triangle.points[:6])
    assert triangle.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(0, 0, 1, 0.5)
        triangle = SmoothTriangle(points=[125, 200, 200, 100, 100, 100, 500, 500, 400, 400])
    r(wid)
    filtered_points = self._filtered_points(triangle.points[:6])
    assert triangle.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(0, 1, 0, 0.5)
        triangle = SmoothTriangle(points=[100, 100, 100.5, 100, 100, 100.5])
    r(wid)
    assert triangle.antialiasing_line_points == []
    triangle.points = [125, 200, 200, 100, 100, 100]
    r(wid)
    assert triangle.antialiasing_line_points == filtered_points + filtered_points[:2]
    triangle.texture = self._get_texture()
    r(wid)
    assert triangle.antialiasing_line_points == []
    triangle.source = ''
    r(wid)
    assert triangle.antialiasing_line_points == filtered_points + filtered_points[:2]