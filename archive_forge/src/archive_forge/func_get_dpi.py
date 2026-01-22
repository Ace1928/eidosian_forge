from os import environ
from kivy.utils import platform
from kivy.properties import AliasProperty
from kivy.event import EventDispatcher
from kivy.setupconfig import USE_SDL2
from kivy.context import register_context
from kivy._metrics import dpi2px, NUMERIC_FORMATS, dispatch_pixel_scale, \
def get_dpi(self, force_recompute=False):
    if not force_recompute and self._dpi is not None:
        return self._dpi
    if platform == 'android':
        if USE_SDL2:
            import jnius
            Hardware = jnius.autoclass('org.renpy.android.Hardware')
            value = Hardware.getDPI()
        else:
            import android
            value = android.get_dpi()
    elif platform == 'ios':
        import ios
        value = ios.get_dpi()
    else:
        from kivy.base import EventLoop
        EventLoop.ensure_window()
        value = EventLoop.window.dpi
    sync_pixel_scale(dpi=value)
    return value