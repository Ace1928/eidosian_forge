import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_aggregate_exceptions_cancels_dependent_tasks_recursively(self):

    def run_tasks_with_exceptions(e1=None, e2=None):
        self.aggregate_exceptions = True
        tasks = (('A', None), ('B', 'A'), ('C', 'B'))
        with self._dep_test(*tasks) as track:
            track.expect_call(1, 'A')
            track.raise_on(1, 'A', e1)
    e1 = Exception('e1')
    exc = self.assertRaises(scheduler.ExceptionGroup, run_tasks_with_exceptions, e1)
    self.assertEqual([e1], exc.exceptions)