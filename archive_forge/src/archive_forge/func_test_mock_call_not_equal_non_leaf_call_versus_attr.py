import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_mock_call_not_equal_non_leaf_call_versus_attr(self):
    m = Mock()
    m.foo.bar()
    self.assertNotEqual(m.mock_calls[0], call.foo().bar())