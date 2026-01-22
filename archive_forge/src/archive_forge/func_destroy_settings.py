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
def destroy_settings(self):
    """.. versionadded:: 1.8.0

        Dereferences the current settings panel if one
        exists. This means that when :meth:`App.open_settings` is next
        run, a new panel will be created and displayed. It doesn't
        affect any of the contents of the panel, but lets you (for
        instance) refresh the settings panel layout if you have
        changed the settings widget in response to a screen size
        change.

        If you have modified :meth:`~App.open_settings` or
        :meth:`~App.display_settings`, you should be careful to
        correctly detect if the previous settings widget has been
        destroyed.

        """
    if self._app_settings is not None:
        self._app_settings = None