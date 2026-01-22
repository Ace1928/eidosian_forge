import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_as_task_timeout_longer(self):
    st = timeutils.wallclock()

    def task():
        while True:
            yield
    self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 0.6, st + 1.5])
    runner = scheduler.TaskRunner(task)
    runner.start(timeout=1)
    self.assertTrue(runner)
    rt = runner.as_task(timeout=10)
    self.assertRaises(scheduler.Timeout, next, rt)
    self.assertEqual(4, timeutils.wallclock.call_count)