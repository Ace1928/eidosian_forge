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
def build_settings(self, settings):
    """.. versionadded:: 1.0.7

        This method is called when the user (or you) want to show the
        application settings. It is called once when the settings panel
        is first opened, after which the panel is cached. It may be
        called again if the cached settings panel is removed by
        :meth:`destroy_settings`.

        You can use this method to add settings panels and to
        customise the settings widget e.g. by changing the sidebar
        width. See the module documentation for full details.

        :Parameters:
            `settings`: :class:`~kivy.uix.settings.Settings`
                Settings instance for adding panels

        """