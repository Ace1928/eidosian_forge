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
def test_awaits_asserts_with_spec_and_any(self):

    class Foo:

        def __eq__(self, other):
            pass
    mock_with_spec = AsyncMock(spec=Foo)

    async def _custom_mock_runnable_test(*args):
        await mock_with_spec(*args)
    run(_custom_mock_runnable_test(Foo(), 1))
    mock_with_spec.assert_has_awaits([call(ANY, 1)])
    mock_with_spec.assert_awaited_with(ANY, 1)
    mock_with_spec.assert_any_await(ANY, 1)