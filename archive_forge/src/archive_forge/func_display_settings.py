from appearing. If you want to prevent the settings instance from appearing
from the same thread and the other coroutines are only executed when Kivy
import os
from inspect import getfile
from os.path import dirname, join, exists, sep, expanduser, isfile
from kivy.config import ConfigParser
from kivy.base import runTouchApp, async_runTouchApp, stopTouchApp
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty
from kivy.setupconfig import USE_SDL2
def display_settings(self, settings):
    """.. versionadded:: 1.8.0

        Display the settings panel. By default, the panel is drawn directly
        on top of the window. You can define other behavior by overriding
        this method, such as adding it to a ScreenManager or Popup.

        You should return True if the display is successful, otherwise False.

        :Parameters:
            `settings`: :class:`~kivy.uix.settings.Settings`
                You can modify this object in order to modify the settings
                display.

        """
    win = self._app_window
    if not win:
        raise Exception('No windows are set on the application, you cannot open settings yet.')
    if settings not in win.children:
        win.add_widget(settings)
        return True
    return False