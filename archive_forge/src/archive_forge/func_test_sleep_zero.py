import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_sleep_zero(self):
    runner = scheduler.TaskRunner(DummyTask())
    runner(wait_time=0)
    self.mock_sleep.assert_called_with(0)