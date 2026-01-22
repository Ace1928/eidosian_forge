import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_misc_multiply(self):
    self.assertEqual(self.call('misc/multiply', a=5, b=2, _rt=int), 10)