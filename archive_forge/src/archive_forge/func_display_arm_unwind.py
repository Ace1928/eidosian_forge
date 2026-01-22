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
def display_arm_unwind(self):
    if not self.elffile.has_ehabi_info():
        self._emitline('There are no .ARM.idx sections in this file.')
        return
    for ehabi_info in self.elffile.get_ehabi_infos():
        self._emitline("\nUnwind section '%s' at offset 0x%x contains %d %s" % (ehabi_info.section_name(), ehabi_info.section_offset(), ehabi_info.num_entry(), 'entry' if ehabi_info.num_entry() == 1 else 'entries'))
        for i in range(ehabi_info.num_entry()):
            entry = ehabi_info.get_entry(i)
            self._emitline()
            self._emitline('Entry %d:' % i)
            if isinstance(entry, CorruptEHABIEntry):
                self._emitline('    [corrupt] %s' % entry.reason)
                continue
            self._emit('    Function offset 0x%x: ' % entry.function_offset)
            if isinstance(entry, CannotUnwindEHABIEntry):
                self._emitline('[cantunwind]')
                continue
            elif entry.eh_table_offset:
                self._emitline('@0x%x' % entry.eh_table_offset)
            else:
                self._emitline('Compact (inline)')
            if isinstance(entry, GenericEHABIEntry):
                self._emitline('    Personality: 0x%x' % entry.personality)
            else:
                self._emitline('    Compact model index: %d' % entry.personality)
                for mnemonic_item in entry.mnmemonic_array():
                    self._emit('    ')
                    self._emitline(mnemonic_item)