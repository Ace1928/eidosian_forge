from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
def on_drop_begin(self, x, y, *args):
    """Event called when a text or a file drop on the application is about
        to begin. It will be followed-up by a single or a multiple
        `on_drop_text` or `on_drop_file` events ending with an `on_drop_end`
        event.

        Arguments `x` and `y` are the mouse cursor position at the time of the
        drop and you should only rely on them if the drop originated from the
        mouse.

        :Parameters:
            `x`: `int`
                Cursor x position, relative to the window :attr:`left`, at the
                time of the drop.
            `y`: `int`
                Cursor y position, relative to the window :attr:`top`, at the
                time of the drop.
            `*args`: `tuple`
                Additional arguments.

        .. note::
            This event works with sdl2 window provider.

        .. versionadded:: 2.1.0
        """
    pass