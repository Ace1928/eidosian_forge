import multiprocessing
import requests
from . import thread
from .._compat import queue
def exceptions(self):
    """Iterate over all the exceptions in the pool.

        :returns: Generator of :class:`~ThreadException`
        """
    while True:
        exc = self.get_exception()
        if exc is None:
            break
        yield exc