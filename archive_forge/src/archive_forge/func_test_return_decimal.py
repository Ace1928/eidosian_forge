import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_decimal(self):
    r = self.call('returntypes/getdecimal')
    self.assertDecimalEquals(r, '3.14159265')