import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_cancel_grace_period(self):
    st = timeutils.wallclock()
    task = DummyTask(5)
    task.do_step = mock.Mock(return_value=None)
    self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 1.0, st + 1.5])
    runner = scheduler.TaskRunner(task)
    self.assertFalse(runner.started())
    runner.start()
    self.assertTrue(runner.started())
    self.assertFalse(runner.step())
    runner.cancel(grace_period=1.0)
    self.assertFalse(runner.step())
    self.assertFalse(runner.step())
    self.assertTrue(runner.step())
    task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])
    self.assertEqual(4, task.do_step.call_count)
    self.mock_sleep.assert_not_called()
    self.assertEqual(4, timeutils.wallclock.call_count)