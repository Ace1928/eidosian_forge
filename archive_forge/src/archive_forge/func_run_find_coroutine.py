import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def run_find_coroutine():
    if myrepl.module_gatherer.find_coroutine():
        main_loop.event_loop.alarm(0, run_find_coroutine)