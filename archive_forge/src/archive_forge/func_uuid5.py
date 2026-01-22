import os
import sys
from enum import Enum, _simple_enum
def uuid5(namespace, name):
    """Generate a UUID from the SHA-1 hash of a namespace UUID and a name."""
    from hashlib import sha1
    hash = sha1(namespace.bytes + bytes(name, 'utf-8')).digest()
    return UUID(bytes=hash[:16], version=5)