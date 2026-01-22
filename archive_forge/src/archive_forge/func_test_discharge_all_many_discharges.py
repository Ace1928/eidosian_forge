import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons.verifier import Verifier
def test_discharge_all_many_discharges(self):
    root_key = b'root key'
    m0 = bakery.Macaroon(root_key=root_key, id=b'id0', location='loc0', version=bakery.LATEST_VERSION)

    class State(object):
        total_required = 40
        id = 1

    def add_caveats(m):
        for i in range(0, 1):
            if State.total_required == 0:
                break
            cid = 'id{}'.format(State.id)
            m.macaroon.add_third_party_caveat(location='somewhere', key='root key {}'.format(cid).encode('utf-8'), key_id=cid.encode('utf-8'))
            State.id += 1
            State.total_required -= 1
    add_caveats(m0)

    def get_discharge(cav, payload):
        self.assertEqual(payload, None)
        m = bakery.Macaroon(root_key='root key {}'.format(cav.caveat_id.decode('utf-8')).encode('utf-8'), id=cav.caveat_id, location='', version=bakery.LATEST_VERSION)
        add_caveats(m)
        return m
    ms = bakery.discharge_all(m0, get_discharge)
    self.assertEqual(len(ms), 41)
    v = Verifier()
    v.satisfy_general(always_ok)
    v.verify(ms[0], root_key, ms[1:])