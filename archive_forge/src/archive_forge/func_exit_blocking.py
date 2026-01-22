import sys
import queue
import threading
from pyglet import app
from pyglet import clock
from pyglet import event
@staticmethod
def exit_blocking():
    """Called by pyglet internal processes when the blocking operation
        completes.  See :py:meth:`enter_blocking`.
        """
    app.platform_event_loop.set_timer(None, None)