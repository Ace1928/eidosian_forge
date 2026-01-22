from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def assert_called(_mock_self):
    """assert that the mock was called at least once
        """
    self = _mock_self
    if self.call_count == 0:
        msg = "Expected '%s' to have been called." % self._mock_name or 'mock'
        raise AssertionError(msg)