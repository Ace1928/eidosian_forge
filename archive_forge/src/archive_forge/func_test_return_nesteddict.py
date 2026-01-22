import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_return_nesteddict(self):
    r = self.call('returntypes/getnesteddict', _rt={wsme.types.bytes: NestedOuter})
    self.assertEqual(r, {b'a': {'inner': {'aint': 0}}, b'b': {'inner': {'aint': 0}}})