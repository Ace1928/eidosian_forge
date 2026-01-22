import argparse
import os, sys
import re
import string
import traceback
import itertools
from elftools import __version__
from elftools.common.exceptions import ELFError
from elftools.common.utils import bytes2str, iterbytes
from elftools.elf.elffile import ELFFile
from elftools.elf.dynamic import DynamicSection, DynamicSegment
from elftools.elf.enums import ENUM_D_TAG
from elftools.elf.segments import InterpSegment
from elftools.elf.sections import (
from elftools.elf.gnuversions import (
from elftools.elf.relocation import RelocationSection
from elftools.elf.descriptions import (
from elftools.elf.constants import E_FLAGS
from elftools.elf.constants import E_FLAGS_MASKS
from elftools.elf.constants import SH_FLAGS
from elftools.elf.constants import SHN_INDICES
from elftools.dwarf.dwarfinfo import DWARFInfo
from elftools.dwarf.descriptions import (
from elftools.dwarf.constants import (
from elftools.dwarf.locationlists import LocationParser, LocationEntry, LocationViewPair, BaseAddressEntry as LocBaseAddressEntry, LocationListsPair
from elftools.dwarf.ranges import RangeEntry, BaseAddressEntry as RangeBaseAddressEntry, RangeListsPair
from elftools.dwarf.callframe import CIE, FDE, ZERO
from elftools.ehabi.ehabiinfo import CorruptEHABIEntry, CannotUnwindEHABIEntry, GenericEHABIEntry
from elftools.dwarf.enums import ENUM_DW_UT
def display_version_info(self):
    """ Display the version info contained in the file
        """
    self._init_versioninfo()
    if not self._versioninfo['type']:
        self._emitline('\nNo version information found in this file.')
        return
    for section in self.elffile.iter_sections():
        if isinstance(section, GNUVerSymSection):
            self._print_version_section_header(section, 'Version symbols')
            num_symbols = section.num_symbols()
            for idx_by_4 in range(0, num_symbols, 4):
                self._emit('  %03x:' % idx_by_4)
                for idx in range(idx_by_4, min(idx_by_4 + 4, num_symbols)):
                    symbol_version = self._symbol_version(idx)
                    if symbol_version['index'] == 'VER_NDX_LOCAL':
                        version_index = 0
                        version_name = '(*local*)'
                    elif symbol_version['index'] == 'VER_NDX_GLOBAL':
                        version_index = 1
                        version_name = '(*global*)'
                    else:
                        version_index = symbol_version['index']
                        version_name = '(%(name)s)' % symbol_version
                    visibility = 'h' if symbol_version['hidden'] else ' '
                    self._emit('%4x%s%-13s' % (version_index, visibility, version_name))
                self._emitline()
        elif isinstance(section, GNUVerDefSection):
            self._print_version_section_header(section, 'Version definition', indent=2)
            offset = 0
            for verdef, verdaux_iter in section.iter_versions():
                verdaux = next(verdaux_iter)
                name = verdaux.name
                if verdef['vd_flags']:
                    flags = describe_ver_flags(verdef['vd_flags'])
                    flags += ' '
                else:
                    flags = 'none'
                self._emitline('  %s: Rev: %i  Flags: %s  Index: %i  Cnt: %i  Name: %s' % (self._format_hex(offset, fieldsize=6, alternate=True), verdef['vd_version'], flags, verdef['vd_ndx'], verdef['vd_cnt'], name))
                verdaux_offset = offset + verdef['vd_aux'] + verdaux['vda_next']
                for idx, verdaux in enumerate(verdaux_iter, start=1):
                    self._emitline('  %s: Parent %i: %s' % (self._format_hex(verdaux_offset, fieldsize=4), idx, verdaux.name))
                    verdaux_offset += verdaux['vda_next']
                offset += verdef['vd_next']
        elif isinstance(section, GNUVerNeedSection):
            self._print_version_section_header(section, 'Version needs')
            offset = 0
            for verneed, verneed_iter in section.iter_versions():
                self._emitline('  %s: Version: %i  File: %s  Cnt: %i' % (self._format_hex(offset, fieldsize=6, alternate=True), verneed['vn_version'], verneed.name, verneed['vn_cnt']))
                vernaux_offset = offset + verneed['vn_aux']
                for idx, vernaux in enumerate(verneed_iter, start=1):
                    if vernaux['vna_flags']:
                        flags = describe_ver_flags(vernaux['vna_flags'])
                        flags += ' '
                    else:
                        flags = 'none'
                    self._emitline('  %s:   Name: %s  Flags: %s  Version: %i' % (self._format_hex(vernaux_offset, fieldsize=4), vernaux.name, flags, vernaux['vna_other']))
                    vernaux_offset += vernaux['vna_next']
                offset += verneed['vn_next']