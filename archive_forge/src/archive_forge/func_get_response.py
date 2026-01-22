import multiprocessing
import requests
from . import thread
from .._compat import queue
def get_response(self):
    """Get a response from the pool.

        :rtype: :class:`~ThreadResponse`
        """
    try:
        request, response = self._response_queue.get_nowait()
    except queue.Empty:
        return None
    else:
        return ThreadResponse(request, response)