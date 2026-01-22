import hashlib
import math
import binascii
from boto.compat import six
def compute_hashes_from_fileobj(fileobj, chunk_size=1024 * 1024):
    """Compute the linear and tree hash from a fileobj.

    This function will compute the linear/tree hash of a fileobj
    in a single pass through the fileobj.

    :param fileobj: A file like object.

    :param chunk_size: The size of the chunks to use for the tree
        hash.  This is also the buffer size used to read from
        `fileobj`.

    :rtype: tuple
    :return: A tuple of (linear_hash, tree_hash).  Both hashes
        are returned in hex.

    """
    if six.PY3 and hasattr(fileobj, 'mode') and ('b' not in fileobj.mode):
        raise ValueError('File-like object must be opened in binary mode!')
    linear_hash = hashlib.sha256()
    chunks = []
    chunk = fileobj.read(chunk_size)
    while chunk:
        if not isinstance(chunk, bytes):
            chunk = chunk.encode(getattr(fileobj, 'encoding', '') or 'utf-8')
        linear_hash.update(chunk)
        chunks.append(hashlib.sha256(chunk).digest())
        chunk = fileobj.read(chunk_size)
    if not chunks:
        chunks = [hashlib.sha256(b'').digest()]
    return (linear_hash.hexdigest(), bytes_to_hex(tree_hash(chunks)))