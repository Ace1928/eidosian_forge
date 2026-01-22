from sys import maxunicode
from typing import Iterable, Iterator, Optional, Set, Tuple, Union
def code_point_order(cp: CodePoint) -> int:
    """Ordering function for code points."""
    return cp if isinstance(cp, int) else cp[0]