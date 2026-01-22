import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_run_as_task_swallow_exception(self):

    class TestException(Exception):
        pass

    def task():
        try:
            yield
        except TestException:
            yield
    tr = scheduler.TaskRunner(task)
    rt = tr.as_task()
    next(rt)
    rt.throw(TestException)
    self.assertFalse(tr.done())
    self.assertRaises(StopIteration, next, rt)
    self.assertTrue(tr.done())