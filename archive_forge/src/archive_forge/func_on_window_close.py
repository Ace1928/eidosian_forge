import sys
import queue
import threading
from pyglet import app
from pyglet import clock
from pyglet import event
def on_window_close(self, window):
    """A window was closed.

            This event is dispatched when a window is closed.  It is not
            dispatched if the window's close button was pressed but the
            window did not close.

            The default handler calls :py:meth:`exit` if no more windows are
            open.  You can override this handler to base your application exit
            on some other policy.

            :event:
            """