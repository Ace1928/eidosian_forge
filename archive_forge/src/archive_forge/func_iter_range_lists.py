import os
from collections import namedtuple
from ..common.utils import struct_parse
from ..common.exceptions import DWARFError
from .dwarf_util import _iter_CUs_in_section
def iter_range_lists(self):
    """ Yields all range lists found in the section according to readelf rules.
        Scans the DIEs for rangelist offsets, then pulls those.
        Returned rangelists are always translated into lists of BaseAddressEntry/RangeEntry objects.
        """
    ver5 = self.version >= 5
    cu_map = {die.attributes['DW_AT_ranges'].value: cu for cu in self._dwarfinfo.iter_CUs() for die in cu.iter_DIEs() if 'DW_AT_ranges' in die.attributes and (cu['version'] >= 5) == ver5}
    all_offsets = list(cu_map.keys())
    all_offsets.sort()
    for offset in all_offsets:
        yield self.get_range_list_at_offset(offset, cu_map[offset])