import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_macaroon_paper_fig6(self):
    """ Implements an example flow as described in the macaroons paper:
        http://theory.stanford.edu/~ataly/Papers/macaroons.pdf
        There are three services, ts, fs, bs:
        ts is a store service which has deligated authority to a forum
        service fs.
        The forum service wants to require its users to be logged into to an
        authentication service bs.

        The client obtains a macaroon from fs (minted by ts, with a third party
         caveat addressed to bs).
        The client obtains a discharge macaroon from bs to satisfy this caveat.
        The target service verifies the original macaroon it delegated to fs
        No direct contact between bs and ts is required
        """
    locator = bakery.ThirdPartyStore()
    bs = common.new_bakery('bs-loc', locator)
    ts = common.new_bakery('ts-loc', locator)
    fs = common.new_bakery('fs-loc', locator)
    ts_macaroon = ts.oven.macaroon(bakery.LATEST_VERSION, common.ages, None, [bakery.LOGIN_OP])
    ts_macaroon.add_caveat(checkers.Caveat(location='bs-loc', condition='user==bob'), fs.oven.key, fs.oven.locator)

    def get_discharge(cav, payload):
        self.assertEqual(cav.location, 'bs-loc')
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, bs.oven.key, common.ThirdPartyStrcmpChecker('user==bob'), bs.oven.locator)
    d = bakery.discharge_all(ts_macaroon, get_discharge)
    ts.checker.auth([d]).allow(common.test_context, [bakery.LOGIN_OP])