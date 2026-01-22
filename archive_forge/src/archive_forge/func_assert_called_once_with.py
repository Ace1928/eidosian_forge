from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def assert_called_once_with(_mock_self, *args, **kwargs):
    """assert that the mock was called exactly once and with the specified
        arguments."""
    self = _mock_self
    if not self.call_count == 1:
        msg = "Expected '%s' to be called once. Called %s times." % (self._mock_name or 'mock', self.call_count)
        raise AssertionError(msg)
    return self.assert_called_with(*args, **kwargs)