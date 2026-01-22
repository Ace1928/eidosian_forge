import hashlib
import math
import binascii
from boto.compat import six
def tree_hash_from_str(str_as_bytes):
    """

    :type str_as_bytes: str
    :param str_as_bytes: The string for which to compute the tree hash.

    :rtype: str
    :return: The computed tree hash, returned as hex.

    """
    return bytes_to_hex(tree_hash(chunk_hashes(str_as_bytes)))