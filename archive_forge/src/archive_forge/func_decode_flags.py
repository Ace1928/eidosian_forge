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
def decode_flags(self, flags):
    description = ''
    if self.elffile['e_machine'] == 'EM_ARM':
        eabi = flags & E_FLAGS.EF_ARM_EABIMASK
        flags &= ~E_FLAGS.EF_ARM_EABIMASK
        if flags & E_FLAGS.EF_ARM_RELEXEC:
            description += ', relocatable executabl'
            flags &= ~E_FLAGS.EF_ARM_RELEXEC
        if eabi == E_FLAGS.EF_ARM_EABI_VER5:
            EF_ARM_KNOWN_FLAGS = E_FLAGS.EF_ARM_ABI_FLOAT_SOFT | E_FLAGS.EF_ARM_ABI_FLOAT_HARD | E_FLAGS.EF_ARM_LE8 | E_FLAGS.EF_ARM_BE8
            description += ', Version5 EABI'
            if flags & E_FLAGS.EF_ARM_ABI_FLOAT_SOFT:
                description += ', soft-float ABI'
            elif flags & E_FLAGS.EF_ARM_ABI_FLOAT_HARD:
                description += ', hard-float ABI'
            if flags & E_FLAGS.EF_ARM_BE8:
                description += ', BE8'
            elif flags & E_FLAGS.EF_ARM_LE8:
                description += ', LE8'
            if flags & ~EF_ARM_KNOWN_FLAGS:
                description += ', <unknown>'
        else:
            description += ', <unrecognized EABI>'
    elif self.elffile['e_machine'] == 'EM_PPC64':
        if flags & E_FLAGS.EF_PPC64_ABI_V2:
            description += ', abiv2'
    elif self.elffile['e_machine'] == 'EM_MIPS':
        if flags & E_FLAGS.EF_MIPS_NOREORDER:
            description += ', noreorder'
        if flags & E_FLAGS.EF_MIPS_PIC:
            description += ', pic'
        if flags & E_FLAGS.EF_MIPS_CPIC:
            description += ', cpic'
        if flags & E_FLAGS.EF_MIPS_ABI2:
            description += ', abi2'
        if flags & E_FLAGS.EF_MIPS_32BITMODE:
            description += ', 32bitmode'
        if flags & E_FLAGS_MASKS.EFM_MIPS_ABI_O32:
            description += ', o32'
        elif flags & E_FLAGS_MASKS.EFM_MIPS_ABI_O64:
            description += ', o64'
        elif flags & E_FLAGS_MASKS.EFM_MIPS_ABI_EABI32:
            description += ', eabi32'
        elif flags & E_FLAGS_MASKS.EFM_MIPS_ABI_EABI64:
            description += ', eabi64'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_1:
            description += ', mips1'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_2:
            description += ', mips2'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_3:
            description += ', mips3'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_4:
            description += ', mips4'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_5:
            description += ', mips5'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_32R2:
            description += ', mips32r2'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_64R2:
            description += ', mips64r2'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_32:
            description += ', mips32'
        if flags & E_FLAGS.EF_MIPS_ARCH == E_FLAGS.EF_MIPS_ARCH_64:
            description += ', mips64'
    elif self.elffile['e_machine'] == 'EM_RISCV':
        if flags & E_FLAGS.EF_RISCV_RVC:
            description += ', RVC'
        if flags & E_FLAGS.EF_RISCV_RVE:
            description += ', RVE'
        if flags & E_FLAGS.EF_RISCV_TSO:
            description += ', TSO'
        if flags & E_FLAGS.EF_RISCV_FLOAT_ABI == E_FLAGS.EF_RISCV_FLOAT_ABI_SOFT:
            description += ', soft-float ABI'
        if flags & E_FLAGS.EF_RISCV_FLOAT_ABI == E_FLAGS.EF_RISCV_FLOAT_ABI_SINGLE:
            description += ', single-float ABI'
        if flags & E_FLAGS.EF_RISCV_FLOAT_ABI == E_FLAGS.EF_RISCV_FLOAT_ABI_DOUBLE:
            description += ', double-float ABI'
        if flags & E_FLAGS.EF_RISCV_FLOAT_ABI == E_FLAGS.EF_RISCV_FLOAT_ABI_QUAD:
            description += ', quad-float ABI'
    elif self.elffile['e_machine'] == 'EM_LOONGARCH':
        if flags & E_FLAGS.EF_LOONGARCH_ABI_MODIFIER_MASK == E_FLAGS.EF_LOONGARCH_ABI_SOFT_FLOAT:
            description += ', SOFT-FLOAT'
        if flags & E_FLAGS.EF_LOONGARCH_ABI_MODIFIER_MASK == E_FLAGS.EF_LOONGARCH_ABI_SINGLE_FLOAT:
            description += ', SINGLE-FLOAT'
        if flags & E_FLAGS.EF_LOONGARCH_ABI_MODIFIER_MASK == E_FLAGS.EF_LOONGARCH_ABI_DOUBLE_FLOAT:
            description += ', DOUBLE-FLOAT'
        if flags & E_FLAGS.EF_LOONGARCH_OBJABI_MASK == E_FLAGS.EF_LOONGARCH_OBJABI_V0:
            description += ', OBJ-v0'
        if flags & E_FLAGS.EF_LOONGARCH_OBJABI_MASK == E_FLAGS.EF_LOONGARCH_OBJABI_V1:
            description += ', OBJ-v1'
    return description