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
def test_spec_parent_not_async_attribute_is(self):

    @patch(async_foo_name, spec=True)
    def test_async(mock_method):
        self.assertIsInstance(mock_method, MagicMock)
        self.assertIsInstance(mock_method.async_method, AsyncMock)
    test_async()