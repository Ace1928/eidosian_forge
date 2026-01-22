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
def test_future_isfuture(self):
    loop = asyncio.new_event_loop()
    fut = loop.create_future()
    loop.stop()
    loop.close()
    mock = AsyncMock(fut)
    self.assertIsInstance(mock, asyncio.Future)