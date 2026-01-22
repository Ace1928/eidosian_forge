import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_start_only(self):
    task = DummyTask()
    task.do_step = mock.Mock(return_value=None)
    runner = scheduler.TaskRunner(task)
    self.assertFalse(runner.started())
    runner.start()
    self.assertTrue(runner.started())
    task.do_step.assert_called_once_with(1)
    self.mock_sleep.assert_not_called()