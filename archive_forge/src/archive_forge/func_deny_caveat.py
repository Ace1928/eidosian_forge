import collections
import pyrfc3339
from ._conditions import (
def deny_caveat(ops):
    """Returns a caveat that will deny attempts to use the macaroon to perform
    any of the listed operations. Operations must not contain a space.
    """
    return _operation_caveat(COND_DENY, ops)