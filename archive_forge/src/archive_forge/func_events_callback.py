from __future__ import annotations
import logging
import os
import threading
import time
import unicodedata
import _watchdog_fsevents as _fsevents  # type: ignore[import-not-found]
from watchdog.events import (
from watchdog.observers.api import DEFAULT_EMITTER_TIMEOUT, DEFAULT_OBSERVER_TIMEOUT, BaseObserver, EventEmitter
from watchdog.utils.dirsnapshot import DirectorySnapshot
def events_callback(self, paths, inodes, flags, ids):
    """Callback passed to FSEventStreamCreate(), it will receive all
        FS events and queue them.
        """
    cls = _fsevents.NativeEvent
    try:
        events = [cls(path, inode, event_flags, event_id) for path, inode, event_flags, event_id in zip(paths, inodes, flags, ids)]
        with self._lock:
            self.queue_events(self.timeout, events)
    except Exception:
        logger.exception('Unhandled exception in fsevents callback')