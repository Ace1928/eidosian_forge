import os
from collections import namedtuple
from bisect import bisect_right
from ..construct.lib.container import Container
from ..common.exceptions import DWARFError
from ..common.utils import (struct_parse, dwarf_assert,
from .structs import DWARFStructs
from .compileunit import CompileUnit
from .abbrevtable import AbbrevTable
from .lineprogram import LineProgram
from .callframe import CallFrameInfo
from .locationlists import LocationLists, LocationListsPair
from .ranges import RangeLists, RangeListsPair
from .aranges import ARanges
from .namelut import NameLUT
from .dwarf_util import _get_base_offset
def get_abbrev_table(self, offset):
    """ Get an AbbrevTable from the given offset in the debug_abbrev
            section.

            The only verification done on the offset is that it's within the
            bounds of the section (if not, an exception is raised).
            It is the caller's responsibility to make sure the offset actually
            points to a valid abbreviation table.

            AbbrevTable objects are cached internally (two calls for the same
            offset will return the same object).
        """
    dwarf_assert(offset < self.debug_abbrev_sec.size, "Offset '0x%x' to abbrev table out of section bounds" % offset)
    if offset not in self._abbrevtable_cache:
        self._abbrevtable_cache[offset] = AbbrevTable(structs=self.structs, stream=self.debug_abbrev_sec.stream, offset=offset)
    return self._abbrevtable_cache[offset]