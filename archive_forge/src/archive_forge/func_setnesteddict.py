import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose({wsme.types.bytes: NestedOuter})
@validate({wsme.types.bytes: NestedOuter})
def setnesteddict(self, value):
    print(repr(value))
    self.assertEqual(type(value), dict)
    self.assertEqual(type(list(value.keys())[0]), wsme.types.bytes)
    self.assertEqual(type(list(value.values())[0]), NestedOuter)
    return value