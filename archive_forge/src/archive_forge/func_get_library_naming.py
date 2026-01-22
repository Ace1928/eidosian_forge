from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
def get_library_naming(self, env: 'Environment', libtype: LibType, strict: bool=False) -> T.Tuple[str, ...]:
    """
        Get library prefixes and suffixes for the target platform ordered by
        priority
        """
    stlibext = ['a']
    if strict and (not isinstance(self, VisualStudioLikeCompiler)):
        prefixes = ['lib']
    else:
        prefixes = ['lib', '']
    if env.machines[self.for_machine].is_darwin():
        shlibext = ['dylib', 'so']
    elif env.machines[self.for_machine].is_windows():
        if isinstance(self, VisualStudioLikeCompiler):
            shlibext = ['lib']
        else:
            shlibext = ['dll.a', 'lib', 'dll']
        stlibext += ['lib']
    elif env.machines[self.for_machine].is_cygwin():
        shlibext = ['dll', 'dll.a']
        prefixes = ['cyg'] + prefixes
    else:
        shlibext = ['so']
    if libtype is LibType.PREFER_SHARED:
        patterns = self._get_patterns(env, prefixes, shlibext, True)
        patterns.extend([x for x in self._get_patterns(env, prefixes, stlibext, False) if x not in patterns])
    elif libtype is LibType.PREFER_STATIC:
        patterns = self._get_patterns(env, prefixes, stlibext, False)
        patterns.extend([x for x in self._get_patterns(env, prefixes, shlibext, True) if x not in patterns])
    elif libtype is LibType.SHARED:
        patterns = self._get_patterns(env, prefixes, shlibext, True)
    else:
        assert libtype is LibType.STATIC
        patterns = self._get_patterns(env, prefixes, stlibext, False)
    return tuple(patterns)