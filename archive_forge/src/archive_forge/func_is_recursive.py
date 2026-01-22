from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
@property
def is_recursive(self):
    """Determines whether subdirectories are watched for the path."""
    return self._is_recursive