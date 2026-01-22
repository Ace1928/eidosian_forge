import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_version1_macaroon_id(self):
    root_key_store = bakery.MemoryKeyStore()
    b = bakery.Bakery(root_key_store=root_key_store, identity_client=common.OneIdentity())
    key, id = root_key_store.root_key()
    root_key_store.get(id)
    m = Macaroon(key=key, version=MACAROON_V1, location='', identifier=id + b'-deadl00f')
    b.checker.auth([[m]]).allow(common.test_context, [bakery.LOGIN_OP])