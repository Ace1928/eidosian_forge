import inspect
from threading import Thread, Event
from kivy.app import App
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.utils import deprecated
def safeOut(self):
    """Provides a thread-safe exit point for interactive launching."""
    self.safe.set()