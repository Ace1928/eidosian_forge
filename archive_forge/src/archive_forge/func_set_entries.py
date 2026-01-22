import os
import collections
from collections import OrderedDict
from collections.abc import Mapping
from ..common.utils import struct_parse
from bisect import bisect_right
import math
from ..construct import CString, Struct, If
def set_entries(self, entries, cu_headers):
    """
        Set the NameLUT entries from an external source. The input is a
        dictionary with the symbol name as the key and NameLUTEntry(cu_ofs,
        die_ofs) as the value.

        This option is useful when dealing with very large ELF files with
        millions of entries. The entries can be parsed once and pickled to a
        file and can be restored via this function on subsequent loads.
        """
    self._entries = entries
    self._cu_headers = cu_headers