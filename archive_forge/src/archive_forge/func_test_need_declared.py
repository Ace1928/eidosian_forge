import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_need_declared(self):
    locator = bakery.ThirdPartyStore()
    first_party = common.new_bakery('first', locator)
    third_party = common.new_bakery('third', locator)
    m = first_party.oven.macaroon(bakery.LATEST_VERSION, common.ages, [checkers.need_declared_caveat(checkers.Caveat(location='third', condition='something'), ['foo', 'bar'])], [bakery.LOGIN_OP])

    def get_discharge(cav, payload):
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, third_party.oven.key, common.ThirdPartyStrcmpChecker('something'), third_party.oven.locator)
    d = bakery.discharge_all(m, get_discharge)
    declared = checkers.infer_declared(d, first_party.checker.namespace())
    self.assertEqual(declared, {'foo': '', 'bar': ''})
    ctx = checkers.context_with_declared(common.test_context, declared)
    first_party.checker.auth([d]).allow(ctx, [bakery.LOGIN_OP])

    def get_discharge(cav, payload):
        checker = common.ThirdPartyCheckerWithCaveats([checkers.declared_caveat('foo', 'a'), checkers.declared_caveat('arble', 'b')])
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, third_party.oven.key, checker, third_party.oven.locator)
    d = bakery.discharge_all(m, get_discharge)
    declared = checkers.infer_declared(d, first_party.checker.namespace())
    self.assertEqual(declared, {'foo': 'a', 'bar': '', 'arble': 'b'})
    ctx = checkers.context_with_declared(common.test_context, declared)
    first_party.checker.auth([d]).allow(ctx, [bakery.LOGIN_OP])

    def get_discharge(cav, payload):
        checker = common.ThirdPartyCheckerWithCaveats([checkers.declared_caveat('foo', 'a'), checkers.declared_caveat('arble', 'b')])
        m = bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, third_party.oven.key, checker, third_party.oven.locator)
        m.add_caveat(checkers.declared_caveat('foo', 'c'), None, None)
        return m
    d = bakery.discharge_all(m, get_discharge)
    declared = checkers.infer_declared(d, first_party.checker.namespace())
    self.assertEqual(declared, {'bar': '', 'arble': 'b'})
    with self.assertRaises(bakery.PermissionDenied) as exc:
        first_party.checker.auth([d]).allow(common.test_context, bakery.LOGIN_OP)
    self.assertEqual('cannot authorize login macaroon: caveat "declared foo a" not satisfied: got foo=null, expected "a"', exc.exception.args[0])