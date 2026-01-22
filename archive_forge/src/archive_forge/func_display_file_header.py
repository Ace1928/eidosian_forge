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
def display_file_header(self):
    """ Display the ELF file header
        """
    self._emitline('ELF Header:')
    self._emit('  Magic:   ')
    self._emit(' '.join(('%2.2x' % b for b in self.elffile.e_ident_raw)))
    self._emitline('      ')
    header = self.elffile.header
    e_ident = header['e_ident']
    self._emitline('  Class:                             %s' % describe_ei_class(e_ident['EI_CLASS']))
    self._emitline('  Data:                              %s' % describe_ei_data(e_ident['EI_DATA']))
    self._emitline('  Version:                           %s' % describe_ei_version(e_ident['EI_VERSION']))
    self._emitline('  OS/ABI:                            %s' % describe_ei_osabi(e_ident['EI_OSABI']))
    self._emitline('  ABI Version:                       %d' % e_ident['EI_ABIVERSION'])
    self._emitline('  Type:                              %s' % describe_e_type(header['e_type'], self.elffile))
    self._emitline('  Machine:                           %s' % describe_e_machine(header['e_machine']))
    self._emitline('  Version:                           %s' % describe_e_version_numeric(header['e_version']))
    self._emitline('  Entry point address:               %s' % self._format_hex(header['e_entry']))
    self._emit('  Start of program headers:          %s' % header['e_phoff'])
    self._emitline(' (bytes into file)')
    self._emit('  Start of section headers:          %s' % header['e_shoff'])
    self._emitline(' (bytes into file)')
    self._emitline('  Flags:                             %s%s' % (self._format_hex(header['e_flags']), self.decode_flags(header['e_flags'])))
    self._emitline('  Size of this header:               %s (bytes)' % header['e_ehsize'])
    self._emitline('  Size of program headers:           %s (bytes)' % header['e_phentsize'])
    self._emitline('  Number of program headers:         %s' % header['e_phnum'])
    self._emitline('  Size of section headers:           %s (bytes)' % header['e_shentsize'])
    self._emit('  Number of section headers:         %s' % header['e_shnum'])
    if header['e_shnum'] == 0 and self.elffile.num_sections() != 0:
        self._emitline(' (%d)' % self.elffile.num_sections())
    else:
        self._emitline('')
    self._emit('  Section header string table index: %s' % header['e_shstrndx'])
    if header['e_shstrndx'] == SHN_INDICES.SHN_XINDEX:
        self._emitline(' (%d)' % self.elffile.get_shstrndx())
    else:
        self._emitline('')