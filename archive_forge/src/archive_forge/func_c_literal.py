from ctypes import c_void_p, c_int, c_bool, Structure, c_uint32, util, cdll, c_uint, c_double, POINTER, c_int64, \
from pyglet.libs.darwin import CFURLRef
def c_literal(literal):
    """Example 'xyz' -> 7895418.
    Used for some CoreAudio constants."""
    num = 0
    for idx, char in enumerate(literal):
        num |= ord(char) << (len(literal) - idx - 1) * 8
    return num