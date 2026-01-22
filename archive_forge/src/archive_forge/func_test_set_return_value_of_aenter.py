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
def test_set_return_value_of_aenter(self):

    def inner_test(mock_type):
        pc = self.ProductionCode()
        pc.session = MagicMock(name='sessionmock')
        cm = mock_type(name='magic_cm')
        response = AsyncMock(name='response')
        response.json = AsyncMock(return_value={'json': 123})
        cm.__aenter__.return_value = response
        pc.session.post.return_value = cm
        result = run(pc.main())
        self.assertEqual(result, {'json': 123})
    for mock_type in [AsyncMock, MagicMock]:
        with self.subTest(f'test set return value of aenter with {mock_type}'):
            inner_test(mock_type)