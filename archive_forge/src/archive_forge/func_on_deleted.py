import os
import signal
import subprocess
import time
from wandb_watchdog.utils import echo, has_attribute
from wandb_watchdog.events import PatternMatchingEventHandler
@echo.echo
def on_deleted(self, event):
    pass