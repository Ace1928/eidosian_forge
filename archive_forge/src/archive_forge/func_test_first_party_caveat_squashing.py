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
def test_first_party_caveat_squashing(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'alice'}})
    ts = _Service('myservice', auth, ids, locator)
    tests = [('duplicates removed', [checkers.Caveat(condition='true 1', namespace='testns'), checkers.Caveat(condition='true 2', namespace='testns'), checkers.Caveat(condition='true 1', namespace='testns'), checkers.Caveat(condition='true 1', namespace='testns'), checkers.Caveat(condition='true 3', namespace='testns')], [checkers.Caveat(condition='true 1', namespace='testns'), checkers.Caveat(condition='true 2', namespace='testns'), checkers.Caveat(condition='true 3', namespace='testns')]), ('earliest time before', [checkers.time_before_caveat(epoch + timedelta(days=1)), checkers.Caveat(condition='true 1', namespace='testns'), checkers.time_before_caveat(epoch + timedelta(days=0, hours=1)), checkers.time_before_caveat(epoch + timedelta(days=0, hours=0, minutes=5))], [checkers.time_before_caveat(epoch + timedelta(days=0, hours=0, minutes=5)), checkers.Caveat(condition='true 1', namespace='testns')]), ('operations and declared caveats removed', [checkers.deny_caveat(['foo']), checkers.allow_caveat(['read', 'write']), checkers.declared_caveat('username', 'bob'), checkers.Caveat(condition='true 1', namespace='testns')], [checkers.Caveat(condition='true 1', namespace='testns')])]
    for test in tests:
        print(test[0])
        ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
        m1 = _Client(locator).capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
        m1.add_caveats(test[1], None, None)
        m2 = _Client(locator).capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
        m2.add_caveat(checkers.Caveat(condition='true notused', namespace='testns'), None, None)
        client = _Client(locator)
        client.add_macaroon(ts, 'authz1', [m1.macaroon])
        client.add_macaroon(ts, 'authz2', [m2.macaroon])
        m3 = client.capability(test_context, ts, [bakery.Op(entity='e1', action='read')])
        self.assertEqual(_macaroon_conditions(m3.macaroon.caveats, False), _resolve_caveats(m3.namespace, test[2]))