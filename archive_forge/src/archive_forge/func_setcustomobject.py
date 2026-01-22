import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose(CustomObject)
@validate(CustomObject)
def setcustomobject(self, value):
    self.assertIsInstance(value, CustomObject)
    self.assertIsInstance(value.name, wsme.types.text)
    self.assertIsInstance(value.aint, int)
    return value