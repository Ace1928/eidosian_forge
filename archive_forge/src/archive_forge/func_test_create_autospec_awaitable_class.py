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
@unittest.skip('Broken test from https://bugs.python.org/issue37251')
def test_create_autospec_awaitable_class(self):
    self.assertIsInstance(create_autospec(AwaitableClass), AsyncMock)