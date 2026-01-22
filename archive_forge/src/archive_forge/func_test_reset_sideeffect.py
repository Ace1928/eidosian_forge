import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_reset_sideeffect(self):
    m = Mock(return_value=10, side_effect=[2, 3])
    m.reset_mock(side_effect=True)
    self.assertEqual(m.return_value, 10)
    self.assertEqual(m.side_effect, None)