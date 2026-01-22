import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_no_shift_cursor_arrow_on_selection(self):
    text = 'some_random_text'
    ti = TextInput(multiline=False, text=text)
    ti.focus = True
    self.render(ti)
    self.assertTrue(ti.focus)
    self.assertEqual(ti.cursor, (len(text), 0))
    steps_skip = 2
    steps_select = 4
    for _ in range(steps_skip):
        ti._key_down((None, None, 'cursor_left', 1), repeat=False)
    ti._key_down((None, None, 'shift', 1), repeat=False)
    for _ in range(steps_select):
        ti._key_down((None, None, 'cursor_left', 1), repeat=False)
    ti._key_up((None, None, 'shift', 1), repeat=False)
    ti._key_down((None, None, 'cursor_right', 1), repeat=False)
    self.assertEqual(ti.cursor, (len(text) - steps_skip, 0))