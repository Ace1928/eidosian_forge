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
def get_ehabi_infos(self):
    """ Generally, shared library and executable contain 1 .ARM.exidx section.
            Object file contains many .ARM.exidx sections.
            So we must traverse every section and filter sections whose type is SHT_ARM_EXIDX.
        """
    _ret = []
    if self['e_type'] == 'ET_REL':
        assert False, "Current version of pyelftools doesn't support relocatable file."
    for section in self.iter_sections(type='SHT_ARM_EXIDX'):
        _ret.append(EHABIInfo(section, self.little_endian))
    return _ret if len(_ret) > 0 else None