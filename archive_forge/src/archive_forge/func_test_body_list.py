import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_body_list(self):
    r = self.call('bodytypes/setlist', body=([10], [int]), _rt=int)
    self.assertEqual(r, 10)