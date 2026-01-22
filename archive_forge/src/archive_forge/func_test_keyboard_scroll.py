import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_keyboard_scroll(self):
    ti = self.make_scrollable_text_input()
    prev_cursor = ti.cursor
    ti.do_cursor_movement('cursor_home', control=True)
    self.advance_frames(1)
    assert ti._visible_lines_range == (0, 10)
    assert prev_cursor != ti.cursor
    prev_cursor = ti.cursor
    ti.do_cursor_movement('cursor_down', control=True)
    self.advance_frames(1)
    assert ti._visible_lines_range == (1, 11)
    assert prev_cursor == ti.cursor
    prev_cursor = ti.cursor
    ti.do_cursor_movement('cursor_up', control=True)
    self.advance_frames(1)
    assert ti._visible_lines_range == (0, 10)
    assert prev_cursor == ti.cursor
    prev_cursor = ti.cursor
    ti.do_cursor_movement('cursor_end', control=True)
    self.advance_frames(1)
    assert ti._visible_lines_range == (20, 30)
    assert prev_cursor != ti.cursor