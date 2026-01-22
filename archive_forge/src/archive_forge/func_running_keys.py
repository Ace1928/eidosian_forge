from eventlet.event import Event
from eventlet import greenthread
import collections
def running_keys(self):
    """
        Return keys for running DAGPool greenthreads. This includes
        greenthreads blocked while iterating through their *results* iterable,
        that is, greenthreads waiting on values from other keys.
        """
    return tuple(self.coros.keys())