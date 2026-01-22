from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
def find_coverage_tools(coredata: coredata.CoreData) -> T.Tuple[T.Optional[str], T.Optional[str], T.Optional[str], T.Optional[str], T.Optional[str], T.Optional[str]]:
    gcovr_exe, gcovr_version = detect_gcovr()
    llvm_cov_exe = detect_llvm_cov(compute_llvm_suffix(coredata))
    lcov_exe, lcov_version, genhtml_exe = detect_lcov_genhtml()
    return (gcovr_exe, gcovr_version, lcov_exe, lcov_version, genhtml_exe, llvm_cov_exe)