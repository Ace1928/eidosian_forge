import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_third_party_discharge_macaroon_ids_are_small(self):
    locator = bakery.ThirdPartyStore()
    bakeries = {'ts-loc': common.new_bakery('ts-loc', locator), 'as1-loc': common.new_bakery('as1-loc', locator), 'as2-loc': common.new_bakery('as2-loc', locator)}
    ts = bakeries['ts-loc']
    ts_macaroon = ts.oven.macaroon(bakery.LATEST_VERSION, common.ages, None, [bakery.LOGIN_OP])
    ts_macaroon.add_caveat(checkers.Caveat(condition='something', location='as1-loc'), ts.oven.key, ts.oven.locator)

    class ThirdPartyCaveatCheckerF(bakery.ThirdPartyCaveatChecker):

        def __init__(self, loc):
            self._loc = loc

        def check_third_party_caveat(self, ctx, info):
            if self._loc == 'as1-loc':
                return [checkers.Caveat(condition='something', location='as2-loc')]
            if self._loc == 'as2-loc':
                return []
            raise bakery.ThirdPartyCaveatCheckFailed('unknown location {}'.format(self._loc))

    def get_discharge(cav, payload):
        oven = bakeries[cav.location].oven
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, oven.key, ThirdPartyCaveatCheckerF(cav.location), oven.locator)
    d = bakery.discharge_all(ts_macaroon, get_discharge)
    ts.checker.auth([d]).allow(common.test_context, [bakery.LOGIN_OP])
    for i, m in enumerate(d):
        for j, cav in enumerate(m.caveats):
            if cav.verification_key_id is not None and len(cav.caveat_id) > 3:
                self.fail('caveat id on caveat {} of macaroon {} is too big ({})'.format(j, i, cav.id))