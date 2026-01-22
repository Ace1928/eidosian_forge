import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setdecimal(self):
    value = decimal.Decimal('3.14')
    assert self.call('argtypes/setdecimal', value=value, _rt=decimal.Decimal) == value