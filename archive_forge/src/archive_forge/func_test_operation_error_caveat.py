from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_operation_error_caveat(self):
    tests = [('empty allow', checkers.allow_caveat(None), 'error no operations allowed'), ('allow: invalid operation name', checkers.allow_caveat(['op1', 'operation number 2']), 'error invalid operation name "operation number 2"'), ('deny: invalid operation name', checkers.deny_caveat(['op1', 'operation number 2']), 'error invalid operation name "operation number 2"')]
    for test in tests:
        print(test[0])
        self.assertEqual(test[1].condition, test[2])