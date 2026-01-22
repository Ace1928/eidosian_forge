import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_run_as_task(self):
    task = DummyTask()
    task.do_step = mock.Mock(return_value=None)
    tr = scheduler.TaskRunner(task)
    rt = tr.as_task()
    for step in rt:
        pass
    self.assertTrue(tr.done())
    task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
    self.assertEqual(3, task.do_step.call_count)
    self.mock_sleep.assert_not_called()