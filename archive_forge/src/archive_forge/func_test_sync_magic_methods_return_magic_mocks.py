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
def test_sync_magic_methods_return_magic_mocks(self):
    a_mock = AsyncMock()
    self.assertIsInstance(a_mock.__enter__, MagicMock)
    self.assertIsInstance(a_mock.__exit__, MagicMock)
    self.assertIsInstance(a_mock.__next__, MagicMock)
    self.assertIsInstance(a_mock.__len__, MagicMock)