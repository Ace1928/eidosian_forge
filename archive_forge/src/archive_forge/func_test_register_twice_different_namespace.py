from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_register_twice_different_namespace(self):
    checker = checkers.Checker()
    checker.namespace().register('testns', '')
    checker.namespace().register('otherns', '')
    checker.register('x', 'testns', lambda x: None)
    with self.assertRaises(checkers.RegisterError) as ctx:
        checker.register('x', 'otherns', lambda x: None)
    self.assertEqual(ctx.exception.args[0], 'checker for x (namespace otherns) already registered in namespace testns')