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
def test_mock_customize_async_context_manager_with_coroutine(self):
    enter_called = False
    exit_called = False

    async def enter_coroutine(*args):
        nonlocal enter_called
        enter_called = True

    async def exit_coroutine(*args):
        nonlocal exit_called
        exit_called = True
    instance = self.WithAsyncContextManager()
    mock_instance = MagicMock(instance)
    mock_instance.__aenter__ = enter_coroutine
    mock_instance.__aexit__ = exit_coroutine

    async def use_context_manager():
        async with mock_instance:
            pass
    run(use_context_manager())
    self.assertTrue(enter_called)
    self.assertTrue(exit_called)