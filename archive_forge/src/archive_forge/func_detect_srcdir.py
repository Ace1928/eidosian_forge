from __future__ import annotations
import glob
import os
import re
import pathlib
import shutil
import subprocess
import typing as T
import functools
from mesonbuild.interpreterbase.decorators import FeatureDeprecated
from .. import mesonlib, mlog
from ..environment import get_llvm_tool_names
from ..mesonlib import version_compare, version_compare_many, search_version, stringlistify, extract_as_list
from .base import DependencyException, DependencyMethods, detect_compiler, strip_system_includedirs, strip_system_libdirs, SystemDependency, ExternalDependency, DependencyTypeName
from .cmake import CMakeDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .misc import threads_factory
from .pkgconfig import PkgConfigDependency
def detect_srcdir(self) -> bool:
    for s in self.src_dirs:
        if os.path.exists(s):
            self.src_dir = s
            self.all_src = mesonlib.File.from_absolute_file(os.path.join(self.src_dir, 'gtest-all.cc'))
            self.main_src = mesonlib.File.from_absolute_file(os.path.join(self.src_dir, 'gtest_main.cc'))
            self.src_include_dirs = [os.path.normpath(os.path.join(self.src_dir, '..')), os.path.normpath(os.path.join(self.src_dir, '../include'))]
            return True
    return False