from __future__ import annotations
import os
import threading
from typing import Callable, Final, cast
from blinker import ANY, Signal
from watchdog import events
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
@classmethod
def get_singleton(cls) -> _MultiPathWatcher:
    """Return the singleton _MultiPathWatcher object.

        Instantiates one if necessary.
        """
    if cls._singleton is None:
        _LOGGER.debug('No singleton. Registering one.')
        _MultiPathWatcher()
    return cast('_MultiPathWatcher', _MultiPathWatcher._singleton)