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
def assert_has_awaits(self, calls, any_order=False):
    """
        Assert the mock has been awaited with the specified calls.
        The :attr:`await_args_list` list is checked for the awaits.

        If `any_order` is False (the default) then the awaits must be
        sequential. There can be extra calls before or after the
        specified awaits.

        If `any_order` is True then the awaits can be in any order, but
        they must all appear in :attr:`await_args_list`.
        """
    expected = [self._call_matcher(c) for c in calls]
    cause = next((e for e in expected if isinstance(e, Exception)), None)
    all_awaits = _CallList((self._call_matcher(c) for c in self.await_args_list))
    if not any_order:
        if expected not in all_awaits:
            if cause is None:
                problem = 'Awaits not found.'
            else:
                problem = 'Error processing expected awaits.\nErrors: {}'.format([e if isinstance(e, Exception) else None for e in expected])
            raise AssertionError(f'{problem}\nExpected: {_CallList(calls)}\nActual: {self.await_args_list}') from cause
        return
    all_awaits = list(all_awaits)
    not_found = []
    for kall in expected:
        try:
            all_awaits.remove(kall)
        except ValueError:
            not_found.append(kall)
    if not_found:
        raise AssertionError('%r not all found in await list' % (tuple(not_found),)) from cause