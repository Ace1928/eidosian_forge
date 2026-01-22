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
def init_async_lib(self, lib):
    super(ClockBaseInterruptBehavior, self).init_async_lib(lib)
    if lib == 'trio':
        import trio
        self._async_event = trio.Event()
        self._async_event.set()
    elif lib == 'asyncio':
        import asyncio
        self._async_event = asyncio.Event()
        self._async_event.set()