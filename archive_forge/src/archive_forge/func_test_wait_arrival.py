import contextlib
import threading
import time
from taskflow import exceptions as excp
from taskflow.persistence.backends import impl_dir
from taskflow import states
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils
def test_wait_arrival(self):
    ev = threading.Event()
    jobs = []

    def poster(wait_post=0.2):
        if not ev.wait(test_utils.WAIT_TIMEOUT):
            raise RuntimeError('Waiter did not appear ready in %s seconds' % test_utils.WAIT_TIMEOUT)
        time.sleep(wait_post)
        self.board.post('test', p_utils.temporary_log_book())

    def waiter():
        ev.set()
        it = self.board.wait()
        jobs.extend(it)
    with connect_close(self.board):
        t1 = threading_utils.daemon_thread(poster)
        t1.start()
        t2 = threading_utils.daemon_thread(waiter)
        t2.start()
        for t in (t1, t2):
            t.join()
    self.assertEqual(1, len(jobs))