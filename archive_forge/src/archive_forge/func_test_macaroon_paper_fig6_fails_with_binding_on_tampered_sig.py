import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_macaroon_paper_fig6_fails_with_binding_on_tampered_sig(self):
    """ Runs a similar test as test_macaroon_paper_fig6 with the discharge
        macaroon binding being done on a tampered signature.
        """
    locator = bakery.ThirdPartyStore()
    bs = common.new_bakery('bs-loc', locator)
    ts = common.new_bakery('ts-loc', locator)
    ts_macaroon = ts.oven.macaroon(bakery.LATEST_VERSION, common.ages, None, [bakery.LOGIN_OP])
    ts_macaroon.add_caveat(checkers.Caveat(condition='user==bob', location='bs-loc'), ts.oven.key, ts.oven.locator)

    def get_discharge(cav, payload):
        self.assertEqual(cav.location, 'bs-loc')
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, bs.oven.key, common.ThirdPartyStrcmpChecker('user==bob'), bs.oven.locator)
    d = bakery.discharge_all(ts_macaroon, get_discharge)
    tampered_macaroon = Macaroon()
    for i, dm in enumerate(d[1:]):
        d[i + 1] = tampered_macaroon.prepare_for_request(dm)
    with self.assertRaises(bakery.PermissionDenied) as exc:
        ts.checker.auth([d]).allow(common.test_context, bakery.LOGIN_OP)
    self.assertEqual('verification failed: Signatures do not match', exc.exception.args[0])