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
def get_aranges(self):
    """ Get an ARanges object representing the .debug_aranges section of
            the DWARF data, or None if the section doesn't exist
        """
    if self.debug_aranges_sec:
        return ARanges(self.debug_aranges_sec.stream, self.debug_aranges_sec.size, self.structs)
    else:
        return None