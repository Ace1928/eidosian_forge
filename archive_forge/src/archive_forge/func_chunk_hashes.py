import hashlib
import math
import binascii
from boto.compat import six
def chunk_hashes(bytestring, chunk_size=_MEGABYTE):
    chunk_count = int(math.ceil(len(bytestring) / float(chunk_size)))
    hashes = []
    for i in range(chunk_count):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        hashes.append(hashlib.sha256(bytestring[start:end]).digest())
    if not hashes:
        return [hashlib.sha256(b'').digest()]
    return hashes