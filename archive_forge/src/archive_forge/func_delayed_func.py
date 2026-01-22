from 1.0.5, you can use a timeout of -1::
from sys import platform
from os import environ
from functools import wraps, partial
from kivy.context import register_context
from kivy.config import Config
from kivy.logger import Logger
from kivy.compat import clock as _default_time
import time
from threading import Event as ThreadingEvent
@wraps(func)
def delayed_func(*args, **kwargs):

    def callback_func(dt):
        func(*args, **kwargs)
    Clock.schedule_once(callback_func, 0)