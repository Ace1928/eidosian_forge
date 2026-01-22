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
@patch(async_foo_name, autospec=True)
def test_normal_method(mock_method):
    self.assertIsInstance(mock_method.normal_method, MagicMock)