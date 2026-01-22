from os import environ
from kivy.utils import platform
from kivy.properties import AliasProperty
from kivy.event import EventDispatcher
from kivy.setupconfig import USE_SDL2
from kivy.context import register_context
from kivy._metrics import dpi2px, NUMERIC_FORMATS, dispatch_pixel_scale, \
def set_density(self, value):
    self._density = value
    sync_pixel_scale(density=value)
    return True