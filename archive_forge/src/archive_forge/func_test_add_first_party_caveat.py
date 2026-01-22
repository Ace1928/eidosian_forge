import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_add_first_party_caveat(self):
    m = bakery.Macaroon('rootkey', 'some id', 'here', bakery.LATEST_VERSION)
    m.add_caveat(checkers.Caveat('test_condition'))
    caveats = m.first_party_caveats()
    self.assertEqual(len(caveats), 1)
    self.assertEqual(caveats[0].caveat_id, b'test_condition')