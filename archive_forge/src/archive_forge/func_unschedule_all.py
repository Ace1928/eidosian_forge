from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
def unschedule_all(self):
    """Unschedules all watches and detaches all associated event
        handlers."""
    with self._lock:
        self._handlers.clear()
        self._clear_emitters()
        self._watches.clear()