import functools
import types
def get_symmetric_key():
    """Returns symmetric key bytes

    16 bytes that were randomly generated. Form a 128 bit key.
    """
    symmetric_key = b'\x92\xcf\x1e\xd9T\xea0p\xd8\xc2H\xae\xc1\xc8r\xa3'
    return symmetric_key