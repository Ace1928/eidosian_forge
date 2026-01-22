import base64
import binascii
import hashlib
import hmac
import io
import re
import struct
import typing
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from spnego._ntlm_raw.des import DES
from spnego._ntlm_raw.md4 import md4
from spnego._ntlm_raw.messages import (
def ntowfv1(password: str) -> bytes:
    """NTLMv1 NTOWFv1 function

    The NT v1 one way function as documented under `NTLM v1 Authentication`_.

    The pseudo-code for this function is::

        Define NTOWFv1(Passwd, User, UserDom) as MD4(UNICODE(Passwd))

    Args:
        password: The password for the user.

    Returns:
        bytes: The NTv1 one way hash of the user's password.

    .. _NTLM v1 Authentication:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/464551a8-9fc4-428e-b3d3-bc5bfb2e73a5
    """
    if is_ntlm_hash(password):
        return base64.b16decode(password.split(':')[1].upper())
    return md4(password.encode('utf-16-le'))