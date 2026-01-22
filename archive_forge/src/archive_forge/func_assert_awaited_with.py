import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
import pkgutil
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock
def assert_awaited_with(self, /, *args, **kwargs):
    """
        Assert that the last await was with the specified arguments.
        """
    if self.await_args is None:
        expected = self._format_mock_call_signature(args, kwargs)
        raise AssertionError(f'Expected await: {expected}\nNot awaited')

    def _error_message():
        msg = self._format_mock_failure_message(args, kwargs, action='await')
        return msg
    expected = self._call_matcher(_Call((args, kwargs), two=True))
    actual = self._call_matcher(self.await_args)
    if actual != expected:
        cause = expected if isinstance(expected, Exception) else None
        raise AssertionError(_error_message()) from cause