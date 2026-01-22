import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
def hasInts(self, only_non_negative=True):
    """Return the number of items in this sequence that are
                integers.

                Args:
                  only_non_negative (boolean):
                    If ``True``, negative integers are not counted in.
                """
    items = [x for x in self._seq if _is_number(x, only_non_negative)]
    return len(items)