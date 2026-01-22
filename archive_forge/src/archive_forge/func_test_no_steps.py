import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
@mock.patch.object(scheduler.TaskRunner, '_sleep')
def test_no_steps(self, mock_sleep):
    self.steps = 0
    with self._dep_test(('second', 'first')):
        pass