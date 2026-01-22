from .abstract import Thenable
from .promises import promise
Barrier.

    Synchronization primitive to call a callback after a list
    of promises have been fulfilled.

    Example:

    .. code-block:: python

        # Request supports the .then() method.
        p1 = http.Request('http://a')
        p2 = http.Request('http://b')
        p3 = http.Request('http://c')
        requests = [p1, p2, p3]

        def all_done():
            pass  # all requests complete

        b = barrier(requests).then(all_done)

        # oops, we forgot we want another request
        b.add(http.Request('http://d'))

    Note that you cannot add new promises to a barrier after
    the barrier is fulfilled.
    