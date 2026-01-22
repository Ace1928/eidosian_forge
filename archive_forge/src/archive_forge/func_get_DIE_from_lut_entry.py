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
def get_DIE_from_lut_entry(self, lut_entry):
    """ Get the DIE from the pubnames or putbtypes lookup table entry.

            lut_entry:
                A NameLUTEntry object from a NameLUT instance (see
                .get_pubmames and .get_pubtypes methods).
        """
    cu = self.get_CU_at(lut_entry.cu_ofs)
    return self.get_DIE_from_refaddr(lut_entry.die_ofs, cu)