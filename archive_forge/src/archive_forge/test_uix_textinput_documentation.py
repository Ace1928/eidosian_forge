import unittest
from itertools import count
from kivy.core.window import Window
from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.textinput import TextInput, TextInputCutCopyPaste
from kivy.uix.widget import Widget
from kivy.clock import Clock
Prepare and start rendering the scrollable text input.

           num_of_lines -- amount of dummy lines used as contents
           n_lines_to_show -- amount of lines to fit in viewport
        