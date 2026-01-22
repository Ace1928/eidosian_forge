import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_selection_enter_multiline(self):
    text = 'multiline\ntext'
    ti = TextInput(multiline=True, text=text)
    ti.focus = True
    self.render(ti)
    self.assertTrue(ti.focus)
    self.assertEqual(ti.cursor, (len(text.split('\n')[-1]), len(text.split('\n')) - 1))
    ti._key_down((None, None, 'shift', 1), repeat=False)
    ti._key_down((None, None, 'cursor_up', 1), repeat=False)
    ti._key_up((None, None, 'shift', 1), repeat=False)
    self.assertEqual(ti.cursor, (len(text.split('\n')[-1]), len(text.split('\n')) - 2))
    self.assertEqual(ti.text, text)
    ti._key_down((None, None, 'enter', 1), repeat=False)
    self.assertEqual(ti.text, text[:4] + '\n')