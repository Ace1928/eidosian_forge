from hashlib import md5
import array
import re
def encode_matrices(matrices):
    """
    Convert a list of 2x2 integer matrices into a sequence of bytes.
    """
    return bytes(array.array('b', sum(sum(matrices, []), [])).tostring())