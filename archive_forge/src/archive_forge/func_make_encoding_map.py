import builtins
import sys
def make_encoding_map(decoding_map):
    """ Creates an encoding map from a decoding map.

        If a target mapping in the decoding map occurs multiple
        times, then that target is mapped to None (undefined mapping),
        causing an exception when encountered by the charmap codec
        during translation.

        One example where this happens is cp875.py which decodes
        multiple character to \\u001a.

    """
    m = {}
    for k, v in decoding_map.items():
        if not v in m:
            m[v] = k
        else:
            m[v] = None
    return m