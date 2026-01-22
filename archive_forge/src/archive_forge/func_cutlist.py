from typing import Optional, List
import numpy as np
from ase.db.core import float_to_time_string, now
def cutlist(lst, length):
    if len(lst) <= length or length == 0:
        return lst
    return lst[:9] + ['... ({} more)'.format(len(lst) - 9)]