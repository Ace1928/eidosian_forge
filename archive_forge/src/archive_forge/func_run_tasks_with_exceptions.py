import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def run_tasks_with_exceptions():
    self.error_wait_time = get_wait_time
    tasks = (('A', None), ('B', None), ('C', 'A'))
    with self._dep_test(*tasks) as track:
        track.expect_call_group(1, ('A', 'B'))
        track.expect_call_group(2, ('A', 'B'))
        track.raise_on(2, 'A', e1)
        track.expect_call(3, 'B')