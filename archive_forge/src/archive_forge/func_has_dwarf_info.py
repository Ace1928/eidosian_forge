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
def has_dwarf_info(self):
    """ Check whether this file appears to have debugging information.
            We assume that if it has the .debug_info or .zdebug_info section, it
            has all the other required sections as well.
        """
    return bool(self.get_section_by_name('.debug_info') or self.get_section_by_name('.zdebug_info') or self.get_section_by_name('.eh_frame'))