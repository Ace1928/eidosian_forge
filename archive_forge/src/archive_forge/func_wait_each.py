from eventlet.event import Event
from eventlet import greenthread
import collections
def wait_each(self, keys=_MISSING):
    """
        *keys* is an optional iterable of keys. If you omit the argument, it
        waits for all the keys from :class:`preload data <DAGPool>`, from
        :meth:`post` calls and from :meth:`spawn` calls: in other words, all
        the keys of which this DAGPool is aware.

        wait_each() is a generator producing (key, value) pairs as a value
        becomes available for each requested key. wait_each() blocks the
        calling greenthread until the next value becomes available. If the
        DAGPool was prepopulated with values for any of the relevant keys, of
        course those can be delivered immediately without waiting.

        Delivery order is intentionally decoupled from the initial sequence of
        keys: each value is delivered as soon as it becomes available. If
        multiple keys are available at the same time, wait_each() delivers
        each of the ready ones in arbitrary order before blocking again.

        The DAGPool does not distinguish between a value returned by one of
        its own greenthreads and one provided by a :meth:`post` call or *preload* data.

        The wait_each() generator terminates (raises StopIteration) when all
        specified keys have been delivered. Thus, typical usage might be:

        ::

            for key, value in dagpool.wait_each(keys):
                # process this ready key and value
            # continue processing now that we've gotten values for all keys

        By implication, if you pass wait_each() an empty iterable of keys, it
        returns immediately without yielding anything.

        If the value to be delivered is a :class:`PropagateError` exception object, the
        generator raises that PropagateError instead of yielding it.

        See also :meth:`wait_each_success`, :meth:`wait_each_exception`.
        """
    return self._wait_each(self._get_keyset_for_wait_each(keys))