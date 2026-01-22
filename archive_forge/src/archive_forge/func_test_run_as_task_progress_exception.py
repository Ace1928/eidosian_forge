import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_run_as_task_progress_exception(self):

    class TestException(Exception):
        pass
    progress_count = []

    def progress():
        if progress_count:
            raise TestException
        progress_count.append(None)
    task = DummyTask()
    task.do_step = mock.Mock(return_value=None)
    tr = scheduler.TaskRunner(task)
    rt = tr.as_task(progress_callback=progress)
    next(rt)
    next(rt)
    self.assertRaises(TestException, next, rt)
    self.assertEqual(1, len(progress_count))
    task.do_step.assert_has_calls([mock.call(1), mock.call(2)])
    self.assertEqual(2, task.do_step.call_count)
    self.mock_sleep.assert_not_called()