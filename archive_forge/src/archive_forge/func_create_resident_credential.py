import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
@classmethod
def create_resident_credential(cls, id: bytes, rp_id: str, user_handle: typing.Optional[bytes], private_key: bytes, sign_count: int) -> 'Credential':
    """Creates a resident (i.e. stateful) credential.

        :Args:
          - id (bytes): Unique base64 encoded string.
          - rp_id (str): Relying party identifier.
          - user_handle (bytes): userHandle associated to the credential. Must be Base64 encoded string.
          - private_key (bytes): Base64 encoded PKCS
          - sign_count (int): intital value for a signature counter.

        :returns:
          - Credential: A resident credential.
        """
    return cls(id, True, rp_id, user_handle, private_key, sign_count)