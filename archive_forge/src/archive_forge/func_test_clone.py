import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_clone(self):
    locator = bakery.ThirdPartyStore()
    bs = common.new_bakery('bs-loc', locator)
    ns = checkers.Namespace({'testns': 'x'})
    m = bakery.Macaroon(root_key=b'root key', id=b'id', location='location', version=bakery.LATEST_VERSION, namespace=ns)
    m.add_caveat(checkers.Caveat(location='bs-loc', condition='something'), bs.oven.key, locator)
    m1 = m.copy()
    self.assertEqual(len(m.macaroon.caveats), 1)
    self.assertEqual(len(m1.macaroon.caveats), 1)
    self.assertEqual(m._caveat_data, m1._caveat_data)
    m.add_caveat(checkers.Caveat(location='bs-loc', condition='something'), bs.oven.key, locator)
    self.assertEqual(len(m.macaroon.caveats), 2)
    self.assertEqual(len(m1.macaroon.caveats), 1)
    self.assertNotEqual(m._caveat_data, m1._caveat_data)