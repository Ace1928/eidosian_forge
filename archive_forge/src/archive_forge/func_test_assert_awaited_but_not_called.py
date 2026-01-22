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
def test_assert_awaited_but_not_called(self):
    with self.assertRaises(AssertionError):
        self.mock.assert_awaited()
    with self.assertRaises(AssertionError):
        self.mock.assert_called()
    with self.assertRaises(TypeError):
        run(self._await_coroutine(self.mock))
    with self.assertRaises(AssertionError):
        self.mock.assert_awaited()
    with self.assertRaises(AssertionError):
        self.mock.assert_called()