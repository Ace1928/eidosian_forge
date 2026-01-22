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
def display_program_headers(self, show_heading=True):
    """ Display the ELF program headers.
            If show_heading is True, displays the heading for this information
            (Elf file type is...)
        """
    self._emitline()
    if self.elffile.num_segments() == 0:
        self._emitline('There are no program headers in this file.')
        return
    elfheader = self.elffile.header
    if show_heading:
        self._emitline('Elf file type is %s' % describe_e_type(elfheader['e_type'], self.elffile))
        self._emitline('Entry point is %s' % self._format_hex(elfheader['e_entry']))
        self._emitline('There are %s program headers, starting at offset %s' % (self.elffile.num_segments(), elfheader['e_phoff']))
        self._emitline()
    self._emitline('Program Headers:')
    if self.elffile.elfclass == 32:
        self._emitline('  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align')
    else:
        self._emitline('  Type           Offset             VirtAddr           PhysAddr')
        self._emitline('                 FileSiz            MemSiz              Flags  Align')
    for segment in self.elffile.iter_segments():
        self._emit('  %-14s ' % describe_p_type(segment['p_type']))
        if self.elffile.elfclass == 32:
            self._emitline('%s %s %s %s %s %-3s %s' % (self._format_hex(segment['p_offset'], fieldsize=6), self._format_hex(segment['p_vaddr'], fullhex=True), self._format_hex(segment['p_paddr'], fullhex=True), self._format_hex(segment['p_filesz'], fieldsize=5), self._format_hex(segment['p_memsz'], fieldsize=5), describe_p_flags(segment['p_flags']), self._format_hex(segment['p_align'])))
        else:
            self._emitline('%s %s %s' % (self._format_hex(segment['p_offset'], fullhex=True), self._format_hex(segment['p_vaddr'], fullhex=True), self._format_hex(segment['p_paddr'], fullhex=True)))
            self._emitline('                 %s %s  %-3s    %s' % (self._format_hex(segment['p_filesz'], fullhex=True), self._format_hex(segment['p_memsz'], fullhex=True), describe_p_flags(segment['p_flags']), self._format_hex(segment['p_align'], lead0x=False)))
        if isinstance(segment, InterpSegment):
            self._emitline('      [Requesting program interpreter: %s]' % segment.get_interp_name())
    if self.elffile.num_sections() == 0:
        return
    self._emitline('\n Section to Segment mapping:')
    self._emitline('  Segment Sections...')
    for nseg, segment in enumerate(self.elffile.iter_segments()):
        self._emit('   %2.2d     ' % nseg)
        for section in self.elffile.iter_sections():
            if not section.is_null() and (not (section['sh_flags'] & SH_FLAGS.SHF_TLS != 0 and section['sh_type'] == 'SHT_NOBITS' and (segment['p_type'] != 'PT_TLS'))) and segment.section_in_segment(section):
                self._emit('%s ' % section.name)
        self._emitline('')