import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_objectarrayattribute(self):
    r = self.call('returntypes/getobjectarrayattribute', _rt=NestedOuter)
    self.assertEqual(r, {'inner': {'aint': 0}, 'inner_array': [{'aint': 12}, {'aint': 13}]})