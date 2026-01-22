from typing import ClassVar, Generic, Optional, Type, TypeVar
import nacl.bindings
from nacl import encoding
from nacl import exceptions as exc
from nacl.encoding import Encoder
from nacl.utils import EncryptedMessage, StringFixer, random
def shared_key(self) -> bytes:
    """
        Returns the Curve25519 shared secret, that can then be used as a key in
        other symmetric ciphers.

        .. warning:: It is **VITALLY** important that you use a nonce with your
            symmetric cipher. If you fail to do this, you compromise the
            privacy of the messages encrypted. Ensure that the key length of
            your cipher is 32 bytes.
        :rtype: [:class:`bytes`]
        """
    return self._shared_key