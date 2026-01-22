import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def compute_key(password: bytes, rev: int, key_size: int, o_entry: bytes, P: int, id1_entry: bytes, metadata_encrypted: bool) -> bytes:
    """
        Algorithm 2: Computing an encryption key.

        a) Pad or truncate the password string to exactly 32 bytes. If the
           password string is more than 32 bytes long,
           use only its first 32 bytes; if it is less than 32 bytes long, pad it
           by appending the required number of
           additional bytes from the beginning of the following padding string:
                < 28 BF 4E 5E 4E 75 8A 41 64 00 4E 56 FF FA 01 08
                2E 2E 00 B6 D0 68 3E 80 2F 0C A9 FE 64 53 69 7A >
           That is, if the password string is n bytes long, append
           the first 32 - n bytes of the padding string to the end
           of the password string. If the password string is empty
           (zero-length), meaning there is no user password,
           substitute the entire padding string in its place.

        b) Initialize the MD5 hash function and pass the result of step (a)
           as input to this function.
        c) Pass the value of the encryption dictionary’s O entry to the
           MD5 hash function. ("Algorithm 3: Computing
           the encryption dictionary’s O (owner password) value" shows how the
           O value is computed.)
        d) Convert the integer value of the P entry to a 32-bit unsigned binary
           number and pass these bytes to the
           MD5 hash function, low-order byte first.
        e) Pass the first element of the file’s file identifier array (the value
           of the ID entry in the document’s trailer
           dictionary; see Table 15) to the MD5 hash function.
        f) (Security handlers of revision 4 or greater) If document metadata is
           not being encrypted, pass 4 bytes with
           the value 0xFFFFFFFF to the MD5 hash function.
        g) Finish the hash.
        h) (Security handlers of revision 3 or greater) Do the following
           50 times: Take the output from the previous
           MD5 hash and pass the first n bytes of the output as input into a new
           MD5 hash, where n is the number of
           bytes of the encryption key as defined by the value of the encryption
           dictionary’s Length entry.
        i) Set the encryption key to the first n bytes of the output from the
           final MD5 hash, where n shall always be 5
           for security handlers of revision 2 but, for security handlers of
           revision 3 or greater, shall depend on the
           value of the encryption dictionary’s Length entry.

        Args:
            password: The encryption secret as a bytes-string
            rev: The encryption revision (see PDF standard)
            key_size: The size of the key in bytes
            o_entry: The owner entry
            P: A set of flags specifying which operations shall be permitted
                when the document is opened with user access. If bit 2 is set to 1,
                all other bits are ignored and all operations are permitted.
                If bit 2 is set to 0, permission for operations are based on the
                values of the remaining flags defined in Table 24.
            id1_entry:
            metadata_encrypted: A boolean indicating if the metadata is encrypted.

        Returns:
            The u_hash digest of length key_size
        """
    a = _padding(password)
    u_hash = hashlib.md5(a)
    u_hash.update(o_entry)
    u_hash.update(struct.pack('<I', P))
    u_hash.update(id1_entry)
    if rev >= 4 and (not metadata_encrypted):
        u_hash.update(b'\xff\xff\xff\xff')
    u_hash_digest = u_hash.digest()
    length = key_size // 8
    if rev >= 3:
        for _ in range(50):
            u_hash_digest = hashlib.md5(u_hash_digest[:length]).digest()
    return u_hash_digest[:length]