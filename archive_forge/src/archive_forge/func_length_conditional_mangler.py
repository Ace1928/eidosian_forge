from hashlib import sha1
from ..util import compat
from ..util import langhelpers
def length_conditional_mangler(length, mangler):
    """a key mangler that mangles if the length of the key is
    past a certain threshold.

    """

    def mangle(key):
        if len(key) >= length:
            return mangler(key)
        else:
            return key
    return mangle