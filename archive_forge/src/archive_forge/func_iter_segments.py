import io
from io import BytesIO
import os
import struct
import zlib
from ..common.exceptions import ELFError, ELFParseError
from ..common.utils import struct_parse, elf_assert
from .structs import ELFStructs
from .sections import (
from .dynamic import DynamicSection, DynamicSegment
from .relocation import (RelocationSection, RelocationHandler,
from .gnuversions import (
from .segments import Segment, InterpSegment, NoteSegment
from ..dwarf.dwarfinfo import DWARFInfo, DebugSectionDescriptor, DwarfConfig
from ..ehabi.ehabiinfo import EHABIInfo
from .hash import ELFHashSection, GNUHashSection
from .constants import SHN_INDICES
def iter_segments(self, type=None):
    """ Yield all the segments in the file. If the optional |type|
            parameter is passed, this method will only yield segments of the
            given type. The parameter value must be a string containing the
            name of the type as defined in the ELF specification, e.g.
            'PT_LOAD'.
        """
    for i in range(self.num_segments()):
        segment = self.get_segment(i)
        if type is None or segment['p_type'] == type:
            yield segment