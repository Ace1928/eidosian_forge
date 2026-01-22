from __future__ import annotations
import binascii
import hashlib
def hash_buffer_hex(buf, hasher=None):
    """
    Same as hash_buffer, but returns its result in hex-encoded form.
    """
    h = hash_buffer(buf, hasher)
    s = binascii.b2a_hex(h)
    return s.decode()