import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def remove_postproc_module(self, mod):
    """Remove a postproc module."""
    if mod in self.postproc_modules:
        self.postproc_modules.remove(mod)