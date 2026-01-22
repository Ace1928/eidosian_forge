import curses
import sys
import threading
from datetime import datetime
from itertools import count
from math import ceil
from textwrap import wrap
from time import time
from celery import VERSION_BANNER, states
from celery.app import app_or_default
from celery.utils.text import abbr, abbrtask
def handle_keypress(self):
    try:
        key = self.win.getkey().upper()
    except Exception:
        return
    key = self.keyalias.get(key) or key
    handler = self.keymap.get(key)
    if handler is not None:
        handler()