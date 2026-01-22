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
def test_auth_with_identity_from_context(self):
    locator = _DischargerLocator()
    ids = _BasicAuthIdService()
    auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'sherlock'}, bakery.Op(entity='e2', action='read'): {'bob'}})
    ts = _Service('myservice', auth, ids, locator)
    ctx = _context_with_basic_auth(test_context, 'sherlock', 'holmes')
    auth_info = _Client(locator).do(ctx, ts, [bakery.Op(entity='e1', action='read')])
    self.assertEqual(auth_info.identity.id(), 'sherlock')
    self.assertEqual(len(auth_info.macaroons), 0)