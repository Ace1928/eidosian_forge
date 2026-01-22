import sys
import threading
from typing import Callable, List, Optional, Type, Union
def switch_and_set(self, new):
    """Switch to a new ``sync_event`` and set the current one.

        Using this method protects against race conditions while setting a new
        ``sync_event``.

        Note that this allows a caller to wait either on the old or the new
        event depending on whether it wants a fine control on what is happening
        inside a thread.

        :param new: The event that will become ``sync_event``
        """
    cur = self.sync_event
    self.lock.acquire()
    try:
        try:
            self.set_sync_event(new)
        except BaseException:
            self.set_sync_event(cur)
            raise
        cur.set()
    finally:
        self.lock.release()