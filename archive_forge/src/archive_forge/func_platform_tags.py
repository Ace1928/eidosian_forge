import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Dict, Generator, Iterator, NamedTuple, Optional, Tuple
from ._elffile import EIClass, EIData, ELFFile, EMachine
def platform_tags(linux: str, arch: str) -> Iterator[str]:
    if not _have_compatible_abi(sys.executable, arch):
        return
    too_old_glibc2 = _GLibCVersion(2, 16)
    if arch in {'x86_64', 'i686'}:
        too_old_glibc2 = _GLibCVersion(2, 4)
    current_glibc = _GLibCVersion(*_get_glibc_version())
    glibc_max_list = [current_glibc]
    for glibc_major in range(current_glibc.major - 1, 1, -1):
        glibc_minor = _LAST_GLIBC_MINOR[glibc_major]
        glibc_max_list.append(_GLibCVersion(glibc_major, glibc_minor))
    for glibc_max in glibc_max_list:
        if glibc_max.major == too_old_glibc2.major:
            min_minor = too_old_glibc2.minor
        else:
            min_minor = -1
        for glibc_minor in range(glibc_max.minor, min_minor, -1):
            glibc_version = _GLibCVersion(glibc_max.major, glibc_minor)
            tag = 'manylinux_{}_{}'.format(*glibc_version)
            if _is_compatible(tag, arch, glibc_version):
                yield linux.replace('linux', tag)
            if glibc_version in _LEGACY_MANYLINUX_MAP:
                legacy_tag = _LEGACY_MANYLINUX_MAP[glibc_version]
                if _is_compatible(legacy_tag, arch, glibc_version):
                    yield linux.replace('linux', legacy_tag)