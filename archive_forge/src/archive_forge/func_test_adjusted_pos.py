import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_adjusted_pos(self):
    from kivy.graphics.boxshadow import BoxShadow
    raw_pos = (50, 50)
    raw_size = (150, 150)
    offset = (10, -100)
    bs = BoxShadow()
    bs.pos = raw_pos
    bs.size = raw_size
    bs.offset = offset
    bs.blur_radius = 80
    bs.spread_radius = (-10, 10)
    adjusted_pos = (raw_pos[0] - bs.blur_radius * 1.5 - bs.spread_radius[0] + bs.offset[0], raw_pos[0] - bs.blur_radius * 1.5 - bs.spread_radius[1] + bs.offset[1])
    assert bs.pos == adjusted_pos
    bs.inset = True
    assert bs.pos == raw_pos
    bs.inset = False
    assert bs.pos == adjusted_pos
    bs = BoxShadow(inset=True, pos=raw_pos, size=raw_size, offset=offset, blur_radius=80, spread_radius=(10, -10))
    adjusted_pos = (raw_pos[0] - bs.blur_radius * 1.5 - bs.spread_radius[0] + bs.offset[0], raw_pos[0] - bs.blur_radius * 1.5 - bs.spread_radius[1] + bs.offset[1])
    assert bs.pos == raw_pos
    bs.inset = False
    assert bs.pos == adjusted_pos
    bs.inset = True
    assert bs.pos == raw_pos