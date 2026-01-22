import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_smoothroundedrectangle(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Color, SmoothRoundedRectangle
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        rounded_rect = SmoothRoundedRectangle(pos=(100, 100), size=(150, 150), radius=[(10, 50), (100, 50), (0, 150), (200, 50)], segments=60)
    r(wid)
    filtered_points = self._filtered_points(rounded_rect.points)
    assert rounded_rect.antialiasing_line_points == filtered_points + filtered_points[:2]
    rounded_rect.size = (150, -2)
    r(wid)
    assert rounded_rect.antialiasing_line_points == []
    rounded_rect.size = (150, 2)
    r(wid)
    assert rounded_rect.antialiasing_line_points == []
    rounded_rect.size = (150, 150)
    r(wid)
    assert rounded_rect.antialiasing_line_points == filtered_points + filtered_points[:2]
    rounded_rect.texture = self._get_texture()
    r(wid)
    assert rounded_rect.antialiasing_line_points == []
    rounded_rect.source = ''
    r(wid)
    assert rounded_rect.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        rounded_rect = SmoothRoundedRectangle(pos=(100, 100), size=(150, 150), segments=0)
    r(wid)
    filtered_points = self._filtered_points(rounded_rect.points)
    assert rounded_rect.antialiasing_line_points == filtered_points + filtered_points[:2]
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        rounded_rect = SmoothRoundedRectangle(pos=(100, 100), size=(150, -3))
    r(wid)
    assert rounded_rect.antialiasing_line_points == []
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1, 0.5)
        rounded_rect = SmoothRoundedRectangle(pos=(100, 100), size=(3.99, 3.99))
    r(wid)
    assert rounded_rect.antialiasing_line_points == []