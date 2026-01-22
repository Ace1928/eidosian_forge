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
def canonical_ops(ops):
    """ Returns the given operations array sorted with duplicates removed.

    @param ops checker.Ops
    @return: checker.Ops
    """
    new_ops = sorted(set(ops), key=lambda x: (x.entity, x.action))
    return new_ops