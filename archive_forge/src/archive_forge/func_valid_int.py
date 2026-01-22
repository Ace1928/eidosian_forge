from typing import NamedTuple
from typing import Sequence, Tuple
def valid_int(s):
    """Returns True if s is a positive integer."""
    return isinstance(s, int) and s > 0