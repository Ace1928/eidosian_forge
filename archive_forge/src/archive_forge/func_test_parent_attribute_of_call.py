import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_parent_attribute_of_call(self):
    self.assertIsNotNone(call.parent)
    self.assertEqual(type(call.parent), _Call)
    self.assertEqual(type(call.parent().parent), _Call)