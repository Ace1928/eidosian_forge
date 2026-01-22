from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def assert_any_call(self, *args, **kwargs):
    """assert the mock has been called with the specified arguments.

        The assert passes if the mock has *ever* been called, unlike
        `assert_called_with` and `assert_called_once_with` that only pass if
        the call is the most recent one."""
    expected = self._call_matcher((args, kwargs))
    actual = [self._call_matcher(c) for c in self.call_args_list]
    if expected not in actual:
        cause = expected if isinstance(expected, Exception) else None
        expected_string = self._format_mock_call_signature(args, kwargs)
        six.raise_from(AssertionError('%s call not found' % expected_string), cause)