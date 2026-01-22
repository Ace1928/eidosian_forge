import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_reset_return(self):
    m = Mock(return_value=10, side_effect=[2, 3])
    m.reset_mock(return_value=True)
    self.assertIsInstance(m.return_value, Mock)
    self.assertNotEqual(m.side_effect, None)