import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_canvas_management(self):
    from kivy.graphics.boxshadow import BoxShadow
    from kivy.uix.widget import Widget
    from kivy.graphics import Color
    r = self.render
    wid = Widget()
    with wid.canvas:
        bs = BoxShadow()
    r(wid)
    assert bs in wid.canvas.children
    wid = Widget()
    bs = BoxShadow()
    wid.canvas.add(Color(1, 0, 0, 1))
    wid.canvas.add(bs)
    r(wid)
    assert bs in wid.canvas.children
    wid.canvas.remove(bs)
    assert bs not in wid.canvas.children
    wid.canvas.insert(1, bs)
    assert bs in wid.canvas.children
    assert wid.canvas.children.index(bs) == 1