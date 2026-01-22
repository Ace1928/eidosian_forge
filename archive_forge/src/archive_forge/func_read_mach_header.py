from __future__ import annotations
import ctypes
import os
import sys
def read_mach_header(lib_file, seek=None):
    """
    This funcition parse mach-O header and extract
    information about minimal system version

    :param lib_file: reference to opened library file with pointer
    """
    if seek is not None:
        lib_file.seek(seek)
    base_class, magic_number = get_base_class_and_magic_number(lib_file)
    arch = '32' if magic_number == MH_MAGIC else '64'

    class SegmentBase(base_class):
        _fields_ = segment_base_fields
    if arch == '32':

        class MachHeader(base_class):
            _fields_ = mach_header_fields
    else:

        class MachHeader(base_class):
            _fields_ = mach_header_fields_64
    mach_header = read_data(MachHeader, lib_file)
    for _i in range(mach_header.ncmds):
        pos = lib_file.tell()
        segment_base = read_data(SegmentBase, lib_file)
        lib_file.seek(pos)
        if segment_base.cmd == LC_VERSION_MIN_MACOSX:

            class VersionMinCommand(base_class):
                _fields_ = version_min_command_fields
            version_info = read_data(VersionMinCommand, lib_file)
            return parse_version(version_info.version)
        elif segment_base.cmd == LC_BUILD_VERSION:

            class VersionBuild(base_class):
                _fields_ = build_version_command_fields
            version_info = read_data(VersionBuild, lib_file)
            return parse_version(version_info.minos)
        else:
            lib_file.seek(pos + segment_base.cmdsize)
            continue