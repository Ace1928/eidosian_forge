from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_register_none_func_raise_exception(self):
    checker = checkers.Checker()
    with self.assertRaises(checkers.RegisterError) as ctx:
        checker.register('x', checkers.STD_NAMESPACE, None)
    self.assertEqual(ctx.exception.args[0], 'no check function registered for namespace std when registering condition x')