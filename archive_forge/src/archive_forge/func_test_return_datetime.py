import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_datetime(self):
    r = self.call('returntypes/getdatetime')
    self.assertDateTimeEquals(r, datetime.datetime(1994, 1, 26, 12))