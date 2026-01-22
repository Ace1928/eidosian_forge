import os
from collections import namedtuple
from ..common.utils import struct_parse
from bisect import bisect_right
import math
def cu_offset_at_addr(self, addr):
    """ Given an address, get the offset of the CU it belongs to, where
            'offset' refers to the offset in the .debug_info section.
        """
    tup = self.entries[bisect_right(self.keys, addr) - 1]
    if tup.begin_addr <= addr < tup.begin_addr + tup.length:
        return tup.info_offset
    else:
        return None