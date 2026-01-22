from __future__ import annotations
import sys, os, subprocess, shutil
import shlex
import typing as T
from .. import envconfig
from .. import mlog
from ..compilers import compilers
from ..compilers.detect import defaults as compiler_names
def deb_detect_cmake(infos: MachineInfo, data: T.Dict[str, str]) -> None:
    system_name_map = {'linux': 'Linux', 'kfreebsd': 'kFreeBSD', 'hurd': 'GNU'}
    system_processor_map = {'arm': 'armv7l', 'mips64el': 'mips64', 'powerpc64le': 'ppc64le'}
    infos.cmake['CMAKE_C_COMPILER'] = infos.compilers['c']
    try:
        infos.cmake['CMAKE_CXX_COMPILER'] = infos.compilers['cpp']
    except KeyError:
        pass
    infos.cmake['CMAKE_SYSTEM_NAME'] = system_name_map[data['DEB_HOST_ARCH_OS']]
    infos.cmake['CMAKE_SYSTEM_PROCESSOR'] = system_processor_map.get(data['DEB_HOST_GNU_CPU'], data['DEB_HOST_GNU_CPU'])