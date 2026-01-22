from __future__ import annotations
import base64
import binascii
import os
import time
import typing
from cryptography import utils
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.hmac import HMAC
def encrypt_at_time(self, msg: bytes, current_time: int) -> bytes:
    return self._fernets[0].encrypt_at_time(msg, current_time)