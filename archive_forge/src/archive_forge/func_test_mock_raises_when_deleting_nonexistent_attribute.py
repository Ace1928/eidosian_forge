import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_raises_when_deleting_nonexistent_attribute(self):
    for mock in (Mock(), MagicMock(), NonCallableMagicMock(), NonCallableMock()):
        del mock.foo
        with self.assertRaises(AttributeError):
            del mock.foo