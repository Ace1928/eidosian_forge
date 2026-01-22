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
def get_pubtypes(self):
    """
        Returns a NameLUT object that contains information read from the
        .debug_pubtypes section in the ELF file.

        NameLUT is essentially a dictionary containing the CU/DIE offsets of
        each symbol. See the NameLUT doc string for more details.
        """
    if self.debug_pubtypes_sec:
        return NameLUT(self.debug_pubtypes_sec.stream, self.debug_pubtypes_sec.size, self.structs)
    else:
        return None