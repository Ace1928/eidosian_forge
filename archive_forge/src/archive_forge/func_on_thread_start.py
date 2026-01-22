from __future__ import with_statement
import os
import threading
from .inotify_buffer import InotifyBuffer
from wandb_watchdog.observers.api import (
from wandb_watchdog.events import (
from wandb_watchdog.utils import unicode_paths
def on_thread_start(self):
    path = unicode_paths.encode(self.watch.path)
    self._inotify = InotifyBuffer(path, self.watch.is_recursive)