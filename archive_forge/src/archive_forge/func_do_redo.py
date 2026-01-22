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
def do_redo(self):
    """Do redo operation.

        .. versionadded:: 1.3.0

        This action re-does any command that has been un-done by
        do_undo/ctrl+z. This function is automatically called when
        `ctrl+r` keys are pressed.
        """
    try:
        x_item = self._redo.pop()
        undo_type = x_item['undo_command'][0]
        _get_cusror_from_index = self.get_cursor_from_index
        if undo_type == 'insert':
            cindex, substring = x_item['redo_command']
            self.cursor = _get_cusror_from_index(cindex)
            self.insert_text(substring, True)
        elif undo_type == 'bkspc':
            self.cursor = _get_cusror_from_index(x_item['redo_command'])
            self.do_backspace(from_undo=True)
        elif undo_type == 'shiftln':
            direction, rows, cursor = x_item['redo_command'][1:]
            self._shift_lines(direction, rows, cursor, True)
        else:
            cindex, scindex = x_item['redo_command']
            self._selection_from = cindex
            self._selection_to = scindex
            self._selection = True
            self.delete_selection(True)
            self.cursor = _get_cusror_from_index(cindex)
        self._undo.append(x_item)
    except IndexError:
        pass