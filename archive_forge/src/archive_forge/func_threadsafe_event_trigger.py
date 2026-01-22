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
def threadsafe_event_trigger(self, event_type: Union[Type[events.Event], Callable[..., None]]) -> Callable[..., None]:
    """Returns a callback to creates events, interrupting current event requests.

        Returned callback function will create an event of type event_type
        which will interrupt an event request if one
        is concurrently occurring, otherwise adding the event to a queue
        that will be checked on the next event request."""
    readfd, writefd = os.pipe()
    self.readers.append(readfd)

    def callback(**kwargs: Any) -> None:
        self.queued_interrupting_events.append(event_type(**kwargs))
        logger.debug('added event to events list %r', self.queued_interrupting_events)
        os.write(writefd, b'interrupting event!')
    return callback