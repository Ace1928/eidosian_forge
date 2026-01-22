import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_settextarray(self):
    value = [u'1']
    r = self.call('argtypes/settextarray', value=(value, [wsme.types.text]), _rt=[wsme.types.text])
    self.assertEqual(r, value)