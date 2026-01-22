import collections
import contextlib
import threading
from fasteners import _utils
import six
@contextlib.contextmanager
def write_lock(self):
    """Context manager that grants a write lock.

        Will wait until no active readers. Blocks readers after acquiring.

        Raises a ``RuntimeError`` if an active reader attempts to acquire
        a lock.
        """
    me = self._current_thread()
    i_am_writer = self.is_writer(check_pending=False)
    if self.is_reader() and (not i_am_writer):
        raise RuntimeError('Reader %s to writer privilege escalation not allowed' % me)
    if i_am_writer:
        yield self
    else:
        with self._cond:
            self._pending_writers.append(me)
            while True:
                if len(self._readers) == 0 and self._writer is None:
                    if self._pending_writers[0] == me:
                        self._writer = self._pending_writers.popleft()
                        break
                self._cond.wait()
        try:
            yield self
        finally:
            with self._cond:
                self._writer = None
                self._cond.notify_all()