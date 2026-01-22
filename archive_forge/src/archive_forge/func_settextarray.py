import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose([wsme.types.text])
@validate([wsme.types.text])
def settextarray(self, value):
    print(repr(value))
    self.assertEqual(type(value), list)
    self.assertEqual(type(value[0]), wsme.types.text)
    return value