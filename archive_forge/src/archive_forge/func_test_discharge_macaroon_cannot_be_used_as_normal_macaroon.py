import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_discharge_macaroon_cannot_be_used_as_normal_macaroon(self):
    locator = bakery.ThirdPartyStore()
    first_party = common.new_bakery('first', locator)
    third_party = common.new_bakery('third', locator)
    m = first_party.oven.macaroon(bakery.LATEST_VERSION, common.ages, [checkers.Caveat(location='third', condition='true')], [bakery.LOGIN_OP])

    class M:
        unbound = None

    def get_discharge(cav, payload):
        m = bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, third_party.oven.key, common.ThirdPartyStrcmpChecker('true'), third_party.oven.locator)
        M.unbound = m.macaroon.copy()
        return m
    bakery.discharge_all(m, get_discharge)
    self.assertIsNotNone(M.unbound)
    with self.assertRaises(bakery.PermissionDenied) as exc:
        third_party.checker.auth([[M.unbound]]).allow(common.test_context, [bakery.LOGIN_OP])
    self.assertEqual('no operations found in macaroon', exc.exception.args[0])