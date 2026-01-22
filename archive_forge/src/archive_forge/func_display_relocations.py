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
def display_relocations(self):
    """ Display the relocations contained in the file
        """
    has_relocation_sections = False
    for section in self.elffile.iter_sections():
        if not isinstance(section, RelocationSection):
            continue
        has_relocation_sections = True
        self._emitline("\nRelocation section '%.128s' at offset %s contains %d %s:" % (section.name, self._format_hex(section['sh_offset']), section.num_relocations(), 'entry' if section.num_relocations() == 1 else 'entries'))
        if section.is_RELA():
            self._emitline('  Offset          Info           Type           Sym. Value    Sym. Name + Addend')
        else:
            self._emitline(' Offset     Info    Type            Sym.Value  Sym. Name')
        symtable = self.elffile.get_section(section['sh_link'])
        for rel in section.iter_relocations():
            hexwidth = 8 if self.elffile.elfclass == 32 else 12
            self._emit('%s  %s %-17.17s' % (self._format_hex(rel['r_offset'], fieldsize=hexwidth, lead0x=False), self._format_hex(rel['r_info'], fieldsize=hexwidth, lead0x=False), describe_reloc_type(rel['r_info_type'], self.elffile)))
            if rel['r_info_sym'] == 0:
                if section.is_RELA():
                    fieldsize = 8 if self.elffile.elfclass == 32 else 16
                    addend = self._format_hex(rel['r_addend'], lead0x=False)
                    self._emit(' %s   %s' % (' ' * fieldsize, addend))
                self._emitline()
            else:
                symbol = symtable.get_symbol(rel['r_info_sym'])
                if symbol['st_name'] == 0:
                    symsecidx = self._get_symbol_shndx(symbol, rel['r_info_sym'], section['sh_link'])
                    symsec = self.elffile.get_section(symsecidx)
                    symbol_name = symsec.name
                    version = ''
                else:
                    symbol_name = symbol.name
                    version = self._symbol_version(rel['r_info_sym'])
                    version = version['name'] if version and version['name'] else ''
                symbol_name = '%.22s' % symbol_name
                if version:
                    symbol_name += '@' + version
                self._emit(' %s %s' % (self._format_hex(symbol['st_value'], fullhex=True, lead0x=False), _format_symbol_name(symbol_name)))
                if section.is_RELA():
                    self._emit(' %s %x' % ('+' if rel['r_addend'] >= 0 else '-', abs(rel['r_addend'])))
                self._emitline()
            if self.elffile.elfclass == 64 and self.elffile['e_machine'] == 'EM_MIPS':
                for i in (2, 3):
                    rtype = rel['r_info_type%s' % i]
                    self._emit('                    Type%s: %s' % (i, describe_reloc_type(rtype, self.elffile)))
                    self._emitline()
    if not has_relocation_sections:
        self._emitline('\nThere are no relocations in this file.')