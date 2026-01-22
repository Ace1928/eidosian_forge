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
def test_operation_allow_caveat(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'bob'}, bakery.Op(entity='e1', action='write'): {'bob'}, bakery.Op(entity='e2', action='read'): {'bob'}})
    ts = _Service('myservice', auth, ids, locator)
    client = _Client(locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    m = client.capability(ctx, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e1', action='write'), bakery.Op(entity='e2', action='read')])
    ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='write')])
    m.add_caveat(checkers.allow_caveat(['read']), None, None)
    ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])
    with self.assertRaises(_DischargeRequiredError):
        ts.do(test_context, [[m.macaroon]], [bakery.Op(entity='e1', action='write')])