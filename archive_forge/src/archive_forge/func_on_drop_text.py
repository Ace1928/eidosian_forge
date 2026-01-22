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
def on_drop_text(self, text, x, y, *args):
    """Event called when a text is dropped on the application.

        Arguments `x` and `y` are the mouse cursor position at the time of the
        drop and you should only rely on them if the drop originated from the
        mouse.

        :Parameters:
            `text`: `bytes`
                Text which is dropped.
            `x`: `int`
                Cursor x position, relative to the window :attr:`left`, at the
                time of the drop.
            `y`: `int`
                Cursor y position, relative to the window :attr:`top`, at the
                time of the drop.
            `*args`: `tuple`
                Additional arguments.

        .. note::
            This event works with sdl2 window provider on x11 window.

        .. note::
            On Windows it is possible to drop a text on the window title bar
            or on its edges and for that case :attr:`mouse_pos` won't be
            updated as the mouse cursor is not within the window.

        .. versionadded:: 2.1.0
        """
    pass