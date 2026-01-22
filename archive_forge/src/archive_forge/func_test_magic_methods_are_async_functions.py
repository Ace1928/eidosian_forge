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
def test_magic_methods_are_async_functions(self):
    m_mock = MagicMock()
    self.assertIsInstance(m_mock.__aenter__, AsyncMock)
    self.assertIsInstance(m_mock.__aexit__, AsyncMock)
    self.assertTrue(iscoroutinefunction(m_mock.__aenter__))
    self.assertTrue(iscoroutinefunction(m_mock.__aexit__))