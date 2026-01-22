import logging
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.delayed_queue import DelayedQueue
from wandb_watchdog.observers.inotify_c import Inotify
def read_event(self):
    """Returns a single event or a tuple of from/to events in case of a
        paired move event. If this buffer has been closed, immediately return
        None.
        """
    return self._queue.get()