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
def set_custom_titlebar(self, widget):
    """
        Sets a Widget as a titlebar

            :widget: The widget you want to set as the titlebar

        .. versionadded:: 2.1.0

        This function returns `True` on successfully setting the custom titlebar,
        else false

        How to use this feature

        ::

            1. first set Window.custom_titlebar to True
            2. then call Window.set_custom_titlebar with the widget/layout you want to set as titlebar as the argument # noqa: E501

        If you want a child of the widget to receive touch events, in
        that child define a property `draggable` and set it to False

        If you set the property `draggable` on a layout,
        all the child in the layout will receive touch events

        If you want to override default behavior, add function `in_drag_area(x,y)`
        to the widget

        The function is call with two args x,y which are mouse.x, and mouse.y
        the function should return

        | `True` if that point should be used to drag the window
        | `False` if you want to receive the touch event at the point

        .. note::
            If you use :meth:`in_drag_area` property `draggable`
            will not be checked

        .. note::
            This feature requires the SDL2 window provider and is currently
            only supported on desktop platforms.

        .. warning::
            :mod:`~kivy.core.window.WindowBase.custom_titlebar` must be set to True
            for the widget to be successfully set as a titlebar

        """
    Logger.warning('Window: set_custom_titlebar is not implemented in the current window provider.')