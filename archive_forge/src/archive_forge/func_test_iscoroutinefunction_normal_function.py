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
def test_iscoroutinefunction_normal_function(self):

    def foo():
        pass
    mock = AsyncMock(foo)
    self.assertTrue(iscoroutinefunction(mock))
    self.assertTrue(inspect.iscoroutinefunction(mock))