import collections
import pyrfc3339
from ._conditions import (
def error_caveat(f):
    """Returns a caveat that will never be satisfied, holding f as the text of
    the caveat.

    This should only be used for highly unusual conditions that are never
    expected to happen in practice, such as a malformed key that is
    conventionally passed as a constant. It's not a panic but you should
    only use it in cases where a panic might possibly be appropriate.

    This mechanism means that caveats can be created without error
    checking and a later systematic check at a higher level (in the
    bakery package) can produce an error instead.
    """
    return _first_party(COND_ERROR, f)