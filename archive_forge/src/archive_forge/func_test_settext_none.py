import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_settext_none(self):
    self.assertEqual(None, self.call('argtypes/settextnone', value=None, _rt=wsme.types.text))