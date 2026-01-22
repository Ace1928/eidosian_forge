import array
import threading
import time
from paramiko.util import b
def read_ready(self):
    """
        Returns true if data is buffered and ready to be read from this
        feeder.  A ``False`` result does not mean that the feeder has closed;
        it means you may need to wait before more data arrives.

        :return:
            ``True`` if a `read` call would immediately return at least one
            byte; ``False`` otherwise.
        """
    self._lock.acquire()
    try:
        if len(self._buffer) == 0:
            return False
        return True
    finally:
        self._lock.release()