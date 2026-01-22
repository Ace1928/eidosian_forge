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
def display_task_row(self, lineno, task):
    state_color = self.state_colors.get(task.state)
    attr = curses.A_NORMAL
    if task.uuid == self.selected_task:
        attr = curses.A_STANDOUT
    timestamp = datetime.utcfromtimestamp(task.timestamp or time())
    timef = timestamp.strftime('%H:%M:%S')
    hostname = task.worker.hostname if task.worker else '*NONE*'
    line = self.format_row(task.uuid, task.name, hostname, timef, task.state)
    self.win.addstr(lineno, LEFT_BORDER_OFFSET, line, attr)
    if state_color:
        self.win.addstr(lineno, len(line) - STATE_WIDTH + BORDER_SPACING - 1, task.state, state_color | attr)