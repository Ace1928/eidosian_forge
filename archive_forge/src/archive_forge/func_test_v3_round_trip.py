import base64
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import nacl.public
import six
from macaroonbakery.bakery import _codec as codec
def test_v3_round_trip(self):
    tp_info = bakery.ThirdPartyInfo(version=bakery.VERSION_3, public_key=self.tp_key.public_key)
    ns = checkers.Namespace()
    ns.register('testns', 'x')
    cid = bakery.encode_caveat('is-authenticated-user', b'a random string', tp_info, self.fp_key, ns)
    res = bakery.decode_caveat(self.tp_key, cid)
    self.assertEqual(res, bakery.ThirdPartyCaveatInfo(first_party_public_key=self.fp_key.public_key, root_key=b'a random string', condition='is-authenticated-user', caveat=cid, third_party_key_pair=self.tp_key, version=bakery.VERSION_3, id=None, namespace=ns))