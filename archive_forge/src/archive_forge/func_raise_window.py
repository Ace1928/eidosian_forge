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
def raise_window(self):
    """Raise the window. This method should be used on desktop
        platforms only.

        .. versionadded:: 1.9.1

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.
        """
    Logger.warning('Window: raise_window is not implemented in the current window provider.')