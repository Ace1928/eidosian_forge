import base64
import hashlib
import itertools
import os
import google
from ._checker import (Op, LOGIN_OP)
from ._store import MemoryKeyStore
from ._error import VerificationError
from ._versions import (
from ._macaroon import (
import macaroonbakery.checkers as checkers
import six
from macaroonbakery._utils import (
from ._internal import id_pb2
from pymacaroons import MACAROON_V2, Verifier
def macaroon(self, version, expiry, caveats, ops):
    """ Takes a macaroon with the given version from the oven,
        associates it with the given operations and attaches the given caveats.
        There must be at least one operation specified.
        The macaroon will expire at the given time - a time_before first party
        caveat will be added with that time.

        @return: a new Macaroon object.
        """
    if len(ops) == 0:
        raise ValueError('cannot mint a macaroon associated with no operations')
    ops = canonical_ops(ops)
    root_key, storage_id = self.root_keystore_for_ops(ops).root_key()
    id = self._new_macaroon_id(storage_id, expiry, ops)
    id_bytes = six.int2byte(LATEST_VERSION) + id.SerializeToString()
    if macaroon_version(version) < MACAROON_V2:
        id_bytes = raw_urlsafe_b64encode(id_bytes)
    m = Macaroon(root_key, id_bytes, self.location, version, self.namespace)
    m.add_caveat(checkers.time_before_caveat(expiry), self.key, self.locator)
    m.add_caveats(caveats, self.key, self.locator)
    return m