import binascii
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.exceptions import MacaroonInvalidSignatureException
from pymacaroons.caveat_delegates import (
from pymacaroons.utils import (
def verify_discharge(self, root, discharge, key, discharge_macaroons=None):
    calculated_signature = hmac_digest(key, discharge.identifier_bytes)
    calculated_signature = self._verify_caveats(root, discharge, discharge_macaroons, calculated_signature)
    if root != discharge:
        calculated_signature = binascii.unhexlify(HashSignaturesBinder(root).bind_signature(binascii.hexlify(calculated_signature)))
    if not self._signatures_match(discharge.signature_bytes, binascii.hexlify(calculated_signature)):
        raise MacaroonInvalidSignatureException('Signatures do not match')
    return True