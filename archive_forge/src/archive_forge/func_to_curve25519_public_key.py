from typing import Optional
import nacl.bindings
from nacl import encoding
from nacl import exceptions as exc
from nacl.public import (
from nacl.utils import StringFixer, random
def to_curve25519_public_key(self) -> _Curve25519_PublicKey:
    """
        Converts a :class:`~nacl.signing.VerifyKey` to a
        :class:`~nacl.public.PublicKey`

        :rtype: :class:`~nacl.public.PublicKey`
        """
    raw_pk = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(self._key)
    return _Curve25519_PublicKey(raw_pk)