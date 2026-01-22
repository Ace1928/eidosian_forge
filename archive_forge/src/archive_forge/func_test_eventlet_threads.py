import os
def test_eventlet_threads(self):
    """Check eventlet compatibility.

            The multiprocessing module is not eventlet friendly and
            must be protected against eventlet thread switching and its
            timeout exceptions.
            """
    th = []
    for i in range(15):
        th.append(eventlet.spawn(self._thread_worker, i % 3, 'abc%d' % i))
    for i in [5, 17, 20, 25]:
        th.append(eventlet.spawn(self._thread_worker_timeout, 2, 'timeout%d' % i, i))
    for thread in th:
        thread.wait()