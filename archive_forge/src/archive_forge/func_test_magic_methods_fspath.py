import math
import unittest
import os
from asyncio import iscoroutinefunction
from unittest.mock import AsyncMock, Mock, MagicMock, _magics
def test_magic_methods_fspath(self):
    mock = MagicMock()
    expected_path = mock.__fspath__()
    mock.reset_mock()
    self.assertEqual(os.fspath(mock), expected_path)
    mock.__fspath__.assert_called_once()