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
def test_assert_awaited(self):
    with self.assertRaises(AssertionError):
        self.mock.assert_awaited()
    run(self._runnable_test())
    self.mock.assert_awaited()