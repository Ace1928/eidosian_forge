from os import environ
from kivy.utils import platform
from kivy.properties import AliasProperty
from kivy.event import EventDispatcher
from kivy.setupconfig import USE_SDL2
from kivy.context import register_context
from kivy._metrics import dpi2px, NUMERIC_FORMATS, dispatch_pixel_scale, \
def get_fontscale(self, force_recompute=False):
    if not force_recompute and self._fontscale is not None:
        return self._fontscale
    value = 1.0
    if platform == 'android':
        from jnius import autoclass
        if USE_SDL2:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
        else:
            PythonActivity = autoclass('org.renpy.android.PythonActivity')
        config = PythonActivity.mActivity.getResources().getConfiguration()
        value = config.fontScale
    sync_pixel_scale(fontscale=value)
    return value