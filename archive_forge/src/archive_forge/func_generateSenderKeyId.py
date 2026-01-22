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
def generateSenderKeyId():
    return KeyHelper.getRandomSequence(2147483647)