import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def test_contains_exceptions(self):
    exception_group = scheduler.ExceptionGroup()
    self.assertIsInstance(exception_group.exceptions, list)