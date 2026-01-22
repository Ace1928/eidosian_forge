import os
from collections import namedtuple
from ..common.utils import struct_parse
from ..common.exceptions import DWARFError
from .dwarf_util import _iter_CUs_in_section
def iter_CU_range_lists_ex(self, cu):
    """For DWARF5, returns untranslated rangelists in the CU, where CU comes from iter_CUs above
        """
    stream = self.stream
    stream.seek(cu.offset_table_offset + (64 if cu.is64 else 32) * cu.offset_count)
    while stream.tell() < cu.offset_after_length + cu.unit_length:
        yield struct_parse(self.structs.Dwarf_rnglists_entries, stream)