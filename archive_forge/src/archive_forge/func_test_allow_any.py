import base64
import json
from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery.tests.common import epoch, test_checker, test_context
from pymacaroons.verifier import FirstPartyCaveatVerifierDelegate, Verifier
def test_allow_any(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}})
    ts = _Service('myservice', auth, ids, locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
    m = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
    client = _Client(locator)
    client.add_macaroon(ts, 'authz', m)
    self._discharges = []
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    with self.assertRaises(_DischargeRequiredError):
        client.do_any(ctx, ts, [bakery.LOGIN_OP, bakery.Op(entity='e1', action='read'), bakery.Op(entity='e1', action='read')])
        self.assertEqual(len(self._discharges), 0)
    _, err = client.do(ctx, ts, [bakery.LOGIN_OP])
    auth_info, allowed = client.do_any(ctx, ts, [bakery.LOGIN_OP, bakery.Op(entity='e1', action='read'), bakery.Op(entity='e1', action='read')])
    self.assertEqual(auth_info.identity.id(), 'bob')
    self.assertEqual(len(auth_info.macaroons), 2)
    self.assertEqual(allowed, [True, True, True])