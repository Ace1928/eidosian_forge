from __future__ import annotations
import functools, json, os, textwrap
from pathlib import Path
import typing as T
from .. import mesonlib, mlog
from .base import process_method_kw, DependencyException, DependencyMethods, DependencyTypeName, ExternalDependency, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from ..environment import detect_cpu_family
from ..programs import ExternalProgram
def find_libpy(self, environment: 'Environment') -> None:
    if self.is_pypy:
        if self.major_version == 3:
            libname = 'pypy3-c'
        else:
            libname = 'pypy-c'
        libdir = os.path.join(self.variables.get('base'), 'bin')
        libdirs = [libdir]
    else:
        libname = f'python{self.version}'
        if 'DEBUG_EXT' in self.variables:
            libname += self.variables['DEBUG_EXT']
        if 'ABIFLAGS' in self.variables:
            libname += self.variables['ABIFLAGS']
        libdirs = []
    largs = self.clib_compiler.find_library(libname, environment, libdirs)
    if largs is not None:
        self.link_args = largs
        self.is_found = True