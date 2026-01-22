from eventlet.event import Event
from eventlet import greenthread
import collections
def wait_each_success(self, keys=_MISSING):
    """
        wait_each_success() filters results so that only success values are
        yielded. In other words, unlike :meth:`wait_each`, wait_each_success()
        will not raise :class:`PropagateError`. Not every provided (or
        defaulted) key will necessarily be represented, though naturally the
        generator will not finish until all have completed.

        In all other respects, wait_each_success() behaves like :meth:`wait_each`.
        """
    for key, value in self._wait_each_raw(self._get_keyset_for_wait_each(keys)):
        if not isinstance(value, PropagateError):
            yield (key, value)