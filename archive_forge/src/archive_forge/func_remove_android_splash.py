import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def remove_android_splash(self, *args):
    """Remove android presplash in SDL2 bootstrap."""
    try:
        from android import remove_presplash
        remove_presplash()
    except ImportError:
        Logger.warning('Base: Failed to import "android" module. Could not remove android presplash.')
        return