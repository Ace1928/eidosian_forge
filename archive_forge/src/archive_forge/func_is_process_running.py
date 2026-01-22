from __future__ import annotations
import functools
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from watchdog.events import EVENT_TYPE_OPENED, FileSystemEvent, PatternMatchingEventHandler
from watchdog.utils import echo
from watchdog.utils.event_debouncer import EventDebouncer
from watchdog.utils.process_watcher import ProcessWatcher
def is_process_running(self):
    return self._process_watchers or (self.process is not None and self.process.poll() is None)