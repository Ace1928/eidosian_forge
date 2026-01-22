import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_binary(self):
    r = self.call('returntypes/getbinary', _rt=wsme.types.binary)
    self.assertEqual(r, binarysample)