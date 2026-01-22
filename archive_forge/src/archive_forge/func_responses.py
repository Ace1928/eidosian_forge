import multiprocessing
import requests
from . import thread
from .._compat import queue
def responses(self):
    """Iterate over all the responses in the pool.

        :returns: Generator of :class:`~ThreadResponse`
        """
    while True:
        resp = self.get_response()
        if resp is None:
            break
        yield resp