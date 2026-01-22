import locale
import logging
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from .termhelpers import Nonblocking
from . import events
from typing import (
from types import TracebackType, FrameType
def scheduled_event_trigger(self, event_type: Type[events.ScheduledEvent]) -> Callable[[float], None]:
    """Returns a callback that schedules events for the future.

        Returned callback function will add an event of type event_type
        to a queue which will be checked the next time an event is requested."""

    def callback(when: float) -> None:
        self.queued_scheduled_events.append((when, event_type(when=when)))
    return callback