import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def test_setnamedattrsobj(self):
    value = {'attr.1': 10, 'attr.2': 20}
    r = self.call('argtypes/setnamedattrsobj', value=(value, NamedAttrsObject), _rt=NamedAttrsObject)
    self.assertEqual(r, value)