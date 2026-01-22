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
def get_dwarf_info(self, relocate_dwarf_sections=True, follow_links=True):
    """ Return a DWARFInfo object representing the debugging information in
            this file.

            If relocate_dwarf_sections is True, relocations for DWARF sections
            are looked up and applied.

            If follow_links is True, we will try to load the supplementary
            object file (if any), and use it to resolve references and imports.
        """
    section_names = ('.debug_info', '.debug_aranges', '.debug_abbrev', '.debug_str', '.debug_line', '.debug_frame', '.debug_loc', '.debug_ranges', '.debug_pubtypes', '.debug_pubnames', '.debug_addr', '.debug_str_offsets', '.debug_line_str', '.debug_loclists', '.debug_rnglists', '.debug_sup', '.gnu_debugaltlink')
    compressed = bool(self.get_section_by_name('.zdebug_info'))
    if compressed:
        section_names = tuple(map(lambda x: '.z' + x[1:], section_names))
    section_names += ('.eh_frame',)
    debug_info_sec_name, debug_aranges_sec_name, debug_abbrev_sec_name, debug_str_sec_name, debug_line_sec_name, debug_frame_sec_name, debug_loc_sec_name, debug_ranges_sec_name, debug_pubtypes_name, debug_pubnames_name, debug_addr_name, debug_str_offsets_name, debug_line_str_name, debug_loclists_sec_name, debug_rnglists_sec_name, debug_sup_name, gnu_debugaltlink_name, eh_frame_sec_name = section_names
    debug_sections = {}
    for secname in section_names:
        section = self.get_section_by_name(secname)
        if section is None:
            debug_sections[secname] = None
        else:
            dwarf_section = self._read_dwarf_section(section, relocate_dwarf_sections)
            if compressed and secname.startswith('.z'):
                dwarf_section = self._decompress_dwarf_section(dwarf_section)
            debug_sections[secname] = dwarf_section
    dwarfinfo = DWARFInfo(config=DwarfConfig(little_endian=self.little_endian, default_address_size=self.elfclass // 8, machine_arch=self.get_machine_arch()), debug_info_sec=debug_sections[debug_info_sec_name], debug_aranges_sec=debug_sections[debug_aranges_sec_name], debug_abbrev_sec=debug_sections[debug_abbrev_sec_name], debug_frame_sec=debug_sections[debug_frame_sec_name], eh_frame_sec=debug_sections[eh_frame_sec_name], debug_str_sec=debug_sections[debug_str_sec_name], debug_loc_sec=debug_sections[debug_loc_sec_name], debug_ranges_sec=debug_sections[debug_ranges_sec_name], debug_line_sec=debug_sections[debug_line_sec_name], debug_pubtypes_sec=debug_sections[debug_pubtypes_name], debug_pubnames_sec=debug_sections[debug_pubnames_name], debug_addr_sec=debug_sections[debug_addr_name], debug_str_offsets_sec=debug_sections[debug_str_offsets_name], debug_line_str_sec=debug_sections[debug_line_str_name], debug_loclists_sec=debug_sections[debug_loclists_sec_name], debug_rnglists_sec=debug_sections[debug_rnglists_sec_name], debug_sup_sec=debug_sections[debug_sup_name], gnu_debugaltlink_sec=debug_sections[gnu_debugaltlink_name])
    if follow_links:
        dwarfinfo.supplementary_dwarfinfo = self.get_supplementary_dwarfinfo(dwarfinfo)
    return dwarfinfo