import hashlib
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional
from pip._internal.exceptions import HashMismatch, HashMissing, InstallationError
from pip._internal.utils.misc import read_chunks
def has_one_of(self, hashes: Dict[str, str]) -> bool:
    """Return whether any of the given hashes are allowed."""
    for hash_name, hex_digest in hashes.items():
        if self.is_hash_allowed(hash_name, hex_digest):
            return True
    return False