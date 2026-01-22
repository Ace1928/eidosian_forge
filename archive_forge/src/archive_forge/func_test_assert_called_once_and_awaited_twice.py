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
def test_assert_called_once_and_awaited_twice(self):
    mock = AsyncMock(AsyncClass)
    coroutine = mock.async_method()
    mock.async_method.assert_called_once()
    run(self._await_coroutine(coroutine))
    with self.assertRaises(RuntimeError):
        run(self._await_coroutine(coroutine))
    mock.async_method.assert_awaited()