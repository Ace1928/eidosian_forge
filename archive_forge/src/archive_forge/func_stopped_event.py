import os
import sys
import threading
from wandb_watchdog.utils import platform
from wandb_watchdog.utils.compat import Event
@property
def stopped_event(self):
    return self._stopped_event