import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_call_repr_loop(self):
    m = Mock()
    m.foo = m
    repr(m.foo())
    self.assertRegex(repr(m.foo()), "<Mock name='mock\\(\\)' id='\\d+'>")