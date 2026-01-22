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
def test_assert_has_mock_calls_on_async_mock_no_spec(self):
    with assertNeverAwaited(self):
        self.mock()
    kalls_empty = [('', (), {})]
    self.assertEqual(self.mock.mock_calls, kalls_empty)
    with assertNeverAwaited(self):
        self.mock('foo')
    with assertNeverAwaited(self):
        self.mock('baz')
    mock_kalls = [call(), call('foo'), call('baz')]
    self.assertEqual(self.mock.mock_calls, mock_kalls)