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
def on_request_close(self, *largs, **kwargs):
    """Event called before we close the window. If a bound function returns
        `True`, the window will not be closed. If the event is triggered
        because of the keyboard escape key, the keyword argument `source` is
        dispatched along with a value of `keyboard` to the bound functions.

        .. warning::
            When the bound function returns True the window will not be closed,
            so use with care because the user would not be able to close the
            program, even if the red X is clicked.
        """
    pass