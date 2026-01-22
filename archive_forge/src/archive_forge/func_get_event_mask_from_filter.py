from __future__ import annotations
import logging
import os
import threading
from typing import Type
from watchdog.events import (
from watchdog.observers.api import DEFAULT_EMITTER_TIMEOUT, DEFAULT_OBSERVER_TIMEOUT, BaseObserver, EventEmitter
from .inotify_buffer import InotifyBuffer
from .inotify_c import InotifyConstants
def get_event_mask_from_filter(self):
    """Optimization: Only include events we are filtering in inotify call"""
    if self._event_filter is None:
        return None
    event_mask = InotifyConstants.IN_DELETE_SELF
    for cls in self._event_filter:
        if cls in (DirMovedEvent, FileMovedEvent):
            event_mask |= InotifyConstants.IN_MOVE
        elif cls in (DirCreatedEvent, FileCreatedEvent):
            event_mask |= InotifyConstants.IN_MOVE | InotifyConstants.IN_CREATE
        elif cls is DirModifiedEvent:
            event_mask |= InotifyConstants.IN_MOVE | InotifyConstants.IN_ATTRIB | InotifyConstants.IN_MODIFY | InotifyConstants.IN_CREATE | InotifyConstants.IN_CLOSE_WRITE
        elif cls is FileModifiedEvent:
            event_mask |= InotifyConstants.IN_ATTRIB | InotifyConstants.IN_MODIFY
        elif cls in (DirDeletedEvent, FileDeletedEvent):
            event_mask |= InotifyConstants.IN_DELETE
        elif cls is FileClosedEvent:
            event_mask |= InotifyConstants.IN_CLOSE
        elif cls is FileOpenedEvent:
            event_mask |= InotifyConstants.IN_OPEN
    return event_mask