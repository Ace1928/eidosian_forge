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
def screenshot(self, name='screenshot{:04d}.png'):
    """Save the actual displayed image to a file.
        """
    i = 0
    path = None
    if name != 'screenshot{:04d}.png':
        _ext = name.split('.')[-1]
        name = ''.join((name[:-(len(_ext) + 1)], '{:04d}.', _ext))
    while True:
        i += 1
        path = join(getcwd(), name.format(i))
        if not exists(path):
            break
    return path