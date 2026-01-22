from base64 import b64encode
from binascii import b2a_base64, a2b_base64
def header_length(bytearray):
    """Return the length of s when it is encoded with base64."""
    groups_of_3, leftover = divmod(len(bytearray), 3)
    n = groups_of_3 * 4
    if leftover:
        n += 4
    return n