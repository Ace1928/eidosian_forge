from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def is_reverse_number_slot(name):
    """
    Tries to identify __radd__ and friends (so the METH_COEXIST flag can be applied).

    There's no great consequence if it inadvertently identifies a few other methods
    so just use a simple rule rather than an exact list.
    """
    if name.startswith('__r') and name.endswith('__'):
        forward_name = name.replace('r', '', 1)
        for meth in get_slot_table(None).PyNumberMethods:
            if hasattr(meth, 'right_slot'):
                return True
    return False