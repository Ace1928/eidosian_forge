from __future__ import annotations
import os.path
import re
import subprocess
import typing as T
from .. import mesonlib
from .. import mlog
from ..arglist import CompilerArgs
from ..linkers import RSPFileSyntax
from ..mesonlib import (
from . import compilers
from .compilers import (
from .mixins.gnu import GnuCompiler
from .mixins.gnu import gnu_common_warning_args
@classmethod
def translate_arg_to_windows(cls, arg: str) -> T.List[str]:
    args: T.List[str] = []
    if arg.startswith('-Wl,'):
        linkargs = arg[arg.index(',') + 1:].split(',')
        for la in linkargs:
            if la.startswith('--out-implib='):
                args.append('-L=/IMPLIB:' + la[13:].strip())
    elif arg.startswith('-mscrtlib='):
        args.append(arg)
        mscrtlib = arg[10:].lower()
        if cls is LLVMDCompiler:
            if mscrtlib != 'libcmt':
                args.append('-L=/NODEFAULTLIB:libcmt')
                args.append('-L=/NODEFAULTLIB:libvcruntime')
            if mscrtlib.startswith('msvcrt'):
                args.append('-L=/DEFAULTLIB:legacy_stdio_definitions.lib')
    return args