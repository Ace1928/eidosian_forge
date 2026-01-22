import time
import eventlet
import testscenarios
import futurist
from futurist.tests import base
from futurist import waiters
def test_wait_for_all(self):
    fs = []
    for _i in range(0, 10):
        fs.append(self.executor.submit(mini_delay, use_eventlet_sleep=self.use_eventlet_sleep))
    done_fs, not_done_fs = waiters.wait_for_all(fs)
    self.assertEqual(len(fs), sum((f.result() for f in done_fs)))
    self.assertEqual(0, len(not_done_fs))