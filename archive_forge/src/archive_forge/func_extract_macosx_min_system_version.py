from __future__ import annotations
import ctypes
import os
import sys
def extract_macosx_min_system_version(path_to_lib):
    with open(path_to_lib, 'rb') as lib_file:
        BaseClass, magic_number = get_base_class_and_magic_number(lib_file, 0)
        if magic_number not in [FAT_MAGIC, FAT_MAGIC_64, MH_MAGIC, MH_MAGIC_64]:
            return
        if magic_number in [FAT_MAGIC, FAT_CIGAM_64]:

            class FatHeader(BaseClass):
                _fields_ = fat_header_fields
            fat_header = read_data(FatHeader, lib_file)
            if magic_number == FAT_MAGIC:

                class FatArch(BaseClass):
                    _fields_ = fat_arch_fields
            else:

                class FatArch(BaseClass):
                    _fields_ = fat_arch_64_fields
            fat_arch_list = [read_data(FatArch, lib_file) for _ in range(fat_header.nfat_arch)]
            versions_list = []
            for el in fat_arch_list:
                try:
                    version = read_mach_header(lib_file, el.offset)
                    if version is not None:
                        if el.cputype == CPU_TYPE_ARM64 and len(fat_arch_list) != 1:
                            if version == (11, 0, 0):
                                continue
                        versions_list.append(version)
                except ValueError:
                    pass
            if len(versions_list) > 0:
                return max(versions_list)
            else:
                return None
        else:
            try:
                return read_mach_header(lib_file, 0)
            except ValueError:
                'when some error during read library files'
                return None