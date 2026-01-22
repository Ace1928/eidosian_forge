import time
import binascii
import os
from random import SystemRandom
from ..ecc.curve import Curve
from ..identitykey import IdentityKey
from ..identitykeypair import IdentityKeyPair
from ..state.prekeyrecord import PreKeyRecord
from ..state.signedprekeyrecord import SignedPreKeyRecord
from .medium import Medium
@staticmethod
def generateRegistrationId(extended_range=False):
    """
        Generate a registration ID.  Clients should only do this once,
        at install time.
        :param extended_range: By default (false), the generated registration ID is sized to require the minimal
        possible protobuf encoding overhead. Specify true if the caller needs the full range of MAX_INT at the cost
        of slightly higher encoding overhead.
        """
    if extended_range:
        regId = KeyHelper.getRandomSequence(2147483646) + 1
    else:
        regId = KeyHelper.getRandomSequence(16380) + 1
    return regId