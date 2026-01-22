from __future__ import unicode_literals
import binascii
from nacl.secret import SecretBox
from pymacaroons import Caveat
from pymacaroons.utils import (
from pymacaroons.exceptions import MacaroonUnmetCaveatException
from .base_third_party import (
def verify_third_party_caveat(self, verifier, caveat, root, macaroon, discharge_macaroons, signature):
    caveat_macaroon = self._caveat_macaroon(caveat, discharge_macaroons)
    caveat_key = self._extract_caveat_key(signature, caveat)
    caveat_met = verifier.verify_discharge(root, caveat_macaroon, caveat_key, discharge_macaroons=discharge_macaroons)
    return caveat_met