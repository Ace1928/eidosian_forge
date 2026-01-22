from sys import maxunicode
from typing import Iterable, Iterator, Optional, Set, Tuple, Union
def get_code_point_range(cp: CodePoint) -> Optional[CodePoint]:
    """
    Returns a code point range.

    :param cp: a single code point or a code point range.
    :return: a code point range or `None` if the argument is not a     code point or a code point range.
    """
    if isinstance(cp, int):
        if 0 <= cp <= maxunicode:
            return (cp, cp + 1)
    else:
        try:
            if isinstance(cp[0], int) and isinstance(cp[1], int):
                if 0 <= cp[0] < cp[1] <= maxunicode + 1:
                    return cp
        except (IndexError, TypeError):
            pass
    return None