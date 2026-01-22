from os.path import join
import sys
from typing import Optional
from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import WindowBase
from kivy.input.provider import MotionEventProvider
from kivy.input.motionevent import MotionEvent
from kivy.resources import resource_find
from kivy.utils import platform, deprecated
from kivy.compat import unichr
from collections import deque
def do_pause(self):
    from kivy.app import App
    from kivy.base import stopTouchApp
    app = App.get_running_app()
    if not app:
        Logger.info('WindowSDL: No running App found, pause.')
    elif not app.dispatch('on_pause'):
        Logger.info("WindowSDL: App doesn't support pause mode, stop.")
        stopTouchApp()
        return
    while True:
        event = self._win.poll()
        if event is False:
            continue
        if event is None:
            continue
        action, args = (event[0], event[1:])
        if action == 'quit':
            EventLoop.quit = True
            break
        elif action == 'app_willenterforeground':
            break
        elif action == 'windowrestored':
            break
    if app:
        app.dispatch('on_resume')