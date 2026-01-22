from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
def test_config_class_args(self):
    TestConfigurable.configure(TestConfig2, b=5)
    obj = cast(TestConfig2, TestConfigurable())
    self.assertIsInstance(obj, TestConfig2)
    self.assertEqual(obj.b, 5)
    obj = cast(TestConfig2, TestConfigurable(42, b=6))
    self.assertIsInstance(obj, TestConfig2)
    self.assertEqual(obj.b, 6)
    self.assertEqual(obj.pos_arg, 42)
    self.checkSubclasses()
    obj = TestConfig2()
    self.assertIs(obj.b, None)