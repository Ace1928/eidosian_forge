import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setbytesarray(self):
    value = [b'1', b'2', b'three']
    r = self.call('argtypes/setbytesarray', value=(value, [wsme.types.bytes]), _rt=[wsme.types.bytes])
    self.assertEqual(r, value)