from os import environ
from kivy.utils import platform
from kivy.properties import AliasProperty
from kivy.event import EventDispatcher
from kivy.setupconfig import USE_SDL2
from kivy.context import register_context
from kivy._metrics import dpi2px, NUMERIC_FORMATS, dispatch_pixel_scale, \
def get_density(self, force_recompute=False):
    if not force_recompute and self._density is not None:
        return self._density
    value = 1.0
    if platform == 'android':
        import jnius
        Hardware = jnius.autoclass('org.renpy.android.Hardware')
        value = Hardware.metrics.scaledDensity
    elif platform == 'ios':
        import ios
        value = ios.get_scale()
    elif platform in ('macosx', 'win'):
        value = self.dpi / 96.0
    sync_pixel_scale(density=value)
    return value