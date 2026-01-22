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
def on_keyboard(self, key, scancode=None, codepoint=None, modifier=None, **kwargs):
    """Event called when keyboard is used.

        .. warning::
            Some providers may omit `scancode`, `codepoint` and/or `modifier`.
        """
    if 'unicode' in kwargs:
        Logger.warning('The use of the unicode parameter is deprecated, and will be removed in future versions. Use codepoint instead, which has identical semantics.')
    is_osx = platform == 'darwin'
    if key == 27 and platform == 'android':
        from android import mActivity
        mActivity.moveTaskToBack(True)
        return True
    elif WindowBase.on_keyboard.exit_on_escape:
        if key == 27 or all([is_osx, key in [113, 119], modifier == 1024]):
            if not self.dispatch('on_request_close', source='keyboard'):
                stopTouchApp()
                self.close()
                return True