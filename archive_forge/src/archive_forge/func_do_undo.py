import re
import sys
import math
from os import environ
from weakref import ref
from itertools import chain, islice
from kivy.animation import Animation
from kivy.base import EventLoop
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.metrics import inch
from kivy.utils import boundary, platform
from kivy.uix.behaviors import FocusBehavior
from kivy.core.text import Label, DEFAULT_FONT
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Callback
from kivy.graphics.context_instructions import Transform
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.bubble import Bubble
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, \
def do_undo(self):
    """Do undo operation.

        .. versionadded:: 1.3.0

        This action un-does any edits that have been made since the last
        call to reset_undo().
        This function is automatically called when `ctrl+z` keys are pressed.
        """
    try:
        x_item = self._undo.pop()
        undo_type = x_item['undo_command'][0]
        self.cursor = self.get_cursor_from_index(x_item['undo_command'][1])
        if undo_type == 'insert':
            cindex, scindex = x_item['undo_command'][1:]
            self._selection_from = cindex
            self._selection_to = scindex
            self._selection = True
            self.delete_selection(True)
        elif undo_type == 'bkspc':
            substring = x_item['undo_command'][2][0]
            mode = x_item['undo_command'][3]
            self.insert_text(substring, True)
            if mode == 'del':
                self.cursor = self.get_cursor_from_index(self.cursor_index() - 1)
        elif undo_type == 'shiftln':
            direction, rows, cursor = x_item['undo_command'][1:]
            self._shift_lines(direction, rows, cursor, True)
        else:
            substring = x_item['undo_command'][2:][0]
            self.insert_text(substring, True)
        self._redo.append(x_item)
        self.scroll_x = self.get_max_scroll_x()
    except IndexError:
        pass