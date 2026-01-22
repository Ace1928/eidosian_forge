import unittest
from ..ecc.curve import Curve
from ..util.keyhelper import KeyHelper
def test_gensig(self):
    identityKeyPair = KeyHelper.generateIdentityKeyPair()
    KeyHelper.generateSignedPreKey(identityKeyPair, 0)