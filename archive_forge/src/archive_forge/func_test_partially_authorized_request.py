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
def test_partially_authorized_request(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}})
    ts = _Service('myservice', auth, ids, locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
    m = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
    client = _Client(locator)
    client.add_macaroon(ts, 'authz', m)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    client.discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])