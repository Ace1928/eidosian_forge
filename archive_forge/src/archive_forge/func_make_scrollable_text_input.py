import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
def make_scrollable_text_input(self, num_of_lines=30, n_lines_to_show=10):
    """Prepare and start rendering the scrollable text input.

           num_of_lines -- amount of dummy lines used as contents
           n_lines_to_show -- amount of lines to fit in viewport
        """
    text = '\n'.join(map(str, range(num_of_lines)))
    ti = TextInput(text=text)
    ti.focus = True
    container = Widget()
    container.add_widget(ti)
    self.render(container)
    ti.height = ti_height_for_x_lines(ti, n_lines_to_show)
    self.advance_frames(1)
    return ti