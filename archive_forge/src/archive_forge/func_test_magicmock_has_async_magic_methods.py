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
def test_magicmock_has_async_magic_methods(self):
    m_mock = MagicMock()
    self.assertTrue(hasattr(m_mock, '__aenter__'))
    self.assertTrue(hasattr(m_mock, '__aexit__'))
    self.assertTrue(hasattr(m_mock, '__anext__'))