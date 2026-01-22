import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_sleep_none(self):
    runner = scheduler.TaskRunner(DummyTask())
    runner(wait_time=None)
    self.mock_sleep.assert_not_called()