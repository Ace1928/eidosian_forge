import logging
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.delayed_queue import DelayedQueue
from wandb_watchdog.observers.inotify_c import Inotify
def matching_from_event(event):
    return not isinstance(event, tuple) and event.is_moved_from and (event.cookie == inotify_event.cookie)