import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_as_task_timeout_shorter(self):
    st = timeutils.wallclock()

    def task():
        while True:
            yield
    self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 0.7, st + 1.6, st + 2.6])
    runner = scheduler.TaskRunner(task)
    runner.start(timeout=10)
    self.assertTrue(runner)
    rt = runner.as_task(timeout=1)
    next(rt)
    self.assertRaises(scheduler.Timeout, next, rt)
    self.assertEqual(5, timeutils.wallclock.call_count)