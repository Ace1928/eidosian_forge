import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_wraps_prevents_automatic_creation_of_mocks(self):

    class Real(object):
        pass
    real = Real()
    mock = Mock(wraps=real)
    self.assertRaises(AttributeError, lambda: mock.new_attr())