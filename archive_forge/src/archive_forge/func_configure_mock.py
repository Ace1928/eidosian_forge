from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def configure_mock(self, **kwargs):
    """Set attributes on the mock through keyword arguments.

        Attributes plus return values and side effects can be set on child
        mocks using standard dot notation and unpacking a dictionary in the
        method call:

        >>> attrs = {'method.return_value': 3, 'other.side_effect': KeyError}
        >>> mock.configure_mock(**attrs)"""
    for arg, val in sorted(kwargs.items(), key=lambda entry: entry[0].count('.')):
        args = arg.split('.')
        final = args.pop()
        obj = self
        for entry in args:
            obj = getattr(obj, entry)
        setattr(obj, final, val)