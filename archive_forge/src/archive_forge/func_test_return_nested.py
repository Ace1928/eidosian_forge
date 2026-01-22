import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_nested(self):
    r = self.call('returntypes/getnested', _rt=NestedOuter)
    self.assertEqual(r, {'inner': {'aint': 0}})