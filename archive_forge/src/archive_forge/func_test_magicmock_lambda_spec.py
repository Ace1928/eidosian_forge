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
def test_magicmock_lambda_spec(self):
    mock_obj = MagicMock()
    mock_obj.mock_func = MagicMock(spec=lambda x: x)
    with patch.object(mock_obj, 'mock_func') as cm:
        self.assertIsInstance(cm, MagicMock)