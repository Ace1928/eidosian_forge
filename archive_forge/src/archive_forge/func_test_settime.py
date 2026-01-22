import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_settime(self):
    value = datetime.time(12, 12, 15)
    r = self.call('argtypes/settime', value=value, _rt=datetime.time)
    self.assertEqual(r, datetime.time(12, 12, 15))