import time
import eventlet
import testscenarios
import futurist
from futurist.tests import base
from futurist import waiters
def test_wait_for_any(self):
    fs = []
    for _i in range(0, 10):
        fs.append(self.executor.submit(mini_delay, use_eventlet_sleep=self.use_eventlet_sleep))
    all_done_fs = []
    total_fs = len(fs)
    while len(all_done_fs) != total_fs:
        done, not_done = waiters.wait_for_any(fs)
        all_done_fs.extend(done)
        fs = not_done
    self.assertEqual(total_fs, sum((f.result() for f in all_done_fs)))