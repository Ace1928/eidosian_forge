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
def test_is_async_function_cm(self):

    def test_async():
        with patch.object(AsyncClass, 'async_method') as mock_method:
            self.assertTrue(iscoroutinefunction(mock_method))
    test_async()