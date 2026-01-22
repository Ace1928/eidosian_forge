import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
@classmethod
def create_non_resident_credential(cls, id: bytes, rp_id: str, private_key: bytes, sign_count: int) -> 'Credential':
    """Creates a non-resident (i.e. stateless) credential.

        :Args:
          - id (bytes): Unique base64 encoded string.
          - rp_id (str): Relying party identifier.
          - private_key (bytes): Base64 encoded PKCS
          - sign_count (int): intital value for a signature counter.

        :Returns:
          - Credential: A non-resident credential.
        """
    return cls(id, False, rp_id, None, private_key, sign_count)