import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def test_cutcopypastebubble_content(self):
    tibubble = TextInputCutCopyPaste(textinput=TextInput())
    assert tibubble.but_copy.parent == tibubble.content
    assert tibubble.but_cut.parent == tibubble.content
    assert tibubble.but_paste.parent == tibubble.content
    assert tibubble.but_selectall.parent == tibubble.content