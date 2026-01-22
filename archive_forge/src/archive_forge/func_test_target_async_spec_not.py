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
def test_target_async_spec_not(self):

    @patch.object(AsyncClass, 'async_method', spec=NormalClass.a)
    def test_async_attribute(mock_method):
        self.assertIsInstance(mock_method, MagicMock)
        self.assertFalse(inspect.iscoroutine(mock_method))
        self.assertFalse(inspect.isawaitable(mock_method))
    test_async_attribute()