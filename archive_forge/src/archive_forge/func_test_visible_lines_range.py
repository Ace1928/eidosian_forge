import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_visible_lines_range(self):
    ti = self.make_scrollable_text_input()
    assert ti._visible_lines_range == (20, 30)
    ti.height = ti_height_for_x_lines(ti, 2.5)
    ti.do_cursor_movement('cursor_home', control=True)
    self.advance_frames(1)
    assert ti._visible_lines_range == (0, 3)
    ti.height = ti_height_for_x_lines(ti, 0)
    self.advance_frames(1)
    assert ti._visible_lines_range == (0, 0)