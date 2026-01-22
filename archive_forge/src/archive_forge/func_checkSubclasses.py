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
def checkSubclasses(self):
    self.assertIsInstance(TestConfig1(), TestConfig1)
    self.assertIsInstance(TestConfig2(), TestConfig2)
    obj = TestConfig1(a=1)
    self.assertEqual(obj.a, 1)
    obj2 = TestConfig2(b=2)
    self.assertEqual(obj2.b, 2)