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
def des(k: bytes, d: bytes) -> bytes:
    """DES encryption.

    Indicates the encryption of an 8-byte data item `d` with the 7-byte key `k` using the Data Encryption Standard
    (DES) algorithm in Electronic Codebook (ECB) mode. The result is 8 bytes in length ([FIPS46-2]).

    Args:
        k: The 7-byte key to use in the DES cipher.
        d: The 8-byte data block to encrypt.

    Returns:
        bytes: The encrypted data block.
    """
    return DES(DES.key56_to_key64(k)).encrypt(d)