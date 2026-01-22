import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_add_third_party_caveat(self):
    locator = bakery.ThirdPartyStore()
    bs = common.new_bakery('bs-loc', locator)
    lbv = six.int2byte(bakery.LATEST_VERSION)
    tests = [('no existing id', b'', [], lbv + six.int2byte(0)), ('several existing ids', b'', [lbv + six.int2byte(0), lbv + six.int2byte(1), lbv + six.int2byte(2)], lbv + six.int2byte(3)), ('with base id', lbv + six.int2byte(0), [lbv + six.int2byte(0)], lbv + six.int2byte(0) + six.int2byte(0)), ('with base id and existing id', lbv + six.int2byte(0), [lbv + six.int2byte(0) + six.int2byte(0)], lbv + six.int2byte(0) + six.int2byte(1))]
    for test in tests:
        print('test ', test[0])
        m = bakery.Macaroon(root_key=b'root key', id=b'id', location='location', version=bakery.LATEST_VERSION)
        for id in test[2]:
            m.macaroon.add_third_party_caveat(key=None, key_id=id, location='')
            m._caveat_id_prefix = test[1]
        m.add_caveat(checkers.Caveat(location='bs-loc', condition='something'), bs.oven.key, locator)
        self.assertEqual(m.macaroon.caveats[len(test[2])].caveat_id, test[3])