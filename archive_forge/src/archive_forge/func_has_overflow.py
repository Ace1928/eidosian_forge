import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once
def has_overflow(grad_norm):
    """
    Detect inf and NaN in grad_norm.
    """
    if grad_norm == float('inf') or grad_norm != grad_norm:
        return True
    return False