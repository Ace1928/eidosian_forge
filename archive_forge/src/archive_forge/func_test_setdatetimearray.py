import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setdatetimearray(self):
    value = [datetime.datetime(2008, 3, 6, 12, 12, 15), datetime.datetime(2008, 4, 6, 2, 12, 15)]
    r = self.call('argtypes/setdatetimearray', value=(value, [datetime.datetime]), _rt=[datetime.datetime])
    self.assertEqual(r, value)