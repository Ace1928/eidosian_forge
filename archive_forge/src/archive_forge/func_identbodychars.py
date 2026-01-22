import sys
from itertools import filterfalse
from typing import List, Tuple, Union
@_lazyclassproperty
def identbodychars(cls):
    """
        all characters in this range that are valid identifier body characters,
        plus the digits 0-9, and · (Unicode MIDDLE DOT)
        """
    return ''.join(sorted(set(cls.identchars + '0123456789·' + ''.join([c for c in cls._chars_for_ranges if ('_' + c).isidentifier()]))))