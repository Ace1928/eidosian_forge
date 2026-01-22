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
def test_async_magic_methods_return_async_mocks(self):
    m_mock = MagicMock()
    self.assertIsInstance(m_mock.__aenter__, AsyncMock)
    self.assertIsInstance(m_mock.__aexit__, AsyncMock)
    self.assertIsInstance(m_mock.__anext__, AsyncMock)
    self.assertIsInstance(m_mock.__aiter__, MagicMock)