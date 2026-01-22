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
def transform_motion_event_2d(self, me, widget=None):
    """Transforms the motion event `me` to this window size and then if
        `widget` is passed transforms `me` to `widget`'s local coordinates.

        :raises:
            `AttributeError`: If widget's ancestor is ``None``.

        .. note::
            Unless it's a specific case, call
            :meth:`~kivy.input.motionevent.MotionEvent.push` before and
            :meth:`~kivy.input.motionevent.MotionEvent.pop` after this method's
            call to preserve previous values of `me`'s attributes.

        .. versionadded:: 2.1.0
        """
    width, height = self._get_effective_size()
    me.scale_for_screen(width, height, rotation=self.rotation, smode=self.softinput_mode, kheight=self.keyboard_height)
    if widget is not None:
        parent = widget.parent
        try:
            if parent:
                me.apply_transform_2d(parent.to_widget)
            else:
                me.apply_transform_2d(widget.to_widget)
                me.apply_transform_2d(widget.to_parent)
        except AttributeError:
            raise