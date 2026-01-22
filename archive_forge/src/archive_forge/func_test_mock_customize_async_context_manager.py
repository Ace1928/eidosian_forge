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
def test_mock_customize_async_context_manager(self):
    instance = self.WithAsyncContextManager()
    mock_instance = MagicMock(instance)
    expected_result = object()
    mock_instance.__aenter__.return_value = expected_result

    async def use_context_manager():
        async with mock_instance as result:
            return result
    self.assertIs(run(use_context_manager()), expected_result)