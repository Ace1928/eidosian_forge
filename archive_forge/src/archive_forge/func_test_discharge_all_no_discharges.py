import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons.verifier import Verifier
def test_discharge_all_no_discharges(self):
    root_key = b'root key'
    m = bakery.Macaroon(root_key=root_key, id=b'id0', location='loc0', version=bakery.LATEST_VERSION, namespace=common.test_checker().namespace())
    ms = bakery.discharge_all(m, no_discharge(self))
    self.assertEqual(len(ms), 1)
    self.assertEqual(ms[0], m.macaroon)
    v = Verifier()
    v.satisfy_general(always_ok)
    v.verify(m.macaroon, root_key, None)