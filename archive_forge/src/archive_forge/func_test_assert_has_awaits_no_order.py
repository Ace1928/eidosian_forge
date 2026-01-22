import asyncio
import gc
import inspect
import re
import unittest
from contextlib import contextmanager
from test import support
from asyncio import run, iscoroutinefunction
from unittest import IsolatedAsyncioTestCase
from unittest.mock import (ANY, call, AsyncMock, patch, MagicMock, Mock,
def test_assert_has_awaits_no_order(self):
    calls = [call('foo'), call('baz')]
    with self.assertRaises(AssertionError) as cm:
        self.mock.assert_has_awaits(calls)
    self.assertEqual(len(cm.exception.args), 1)
    run(self._runnable_test('foo'))
    with self.assertRaises(AssertionError):
        self.mock.assert_has_awaits(calls)
    run(self._runnable_test('foo'))
    with self.assertRaises(AssertionError):
        self.mock.assert_has_awaits(calls)
    run(self._runnable_test('baz'))
    self.mock.assert_has_awaits(calls)
    run(self._runnable_test('SomethingElse'))
    self.mock.assert_has_awaits(calls)