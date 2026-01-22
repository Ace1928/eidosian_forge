from __future__ import annotations
from ..mesonlib import (
from ..envconfig import BinaryTable
from .. import mlog
from ..linkers import guess_win_linker, guess_nix_linker
import subprocess
import platform
import re
import shutil
import tempfile
import os
import typing as T
def detect_swift_compiler(env: 'Environment', for_machine: MachineChoice) -> Compiler:
    from .swift import SwiftCompiler
    exelist = env.lookup_binary_entry(for_machine, 'swift')
    is_cross = env.is_cross_build(for_machine)
    info = env.machines[for_machine]
    if exelist is None:
        exelist = [defaults['swift'][0]]
    try:
        p, _, err = Popen_safe_logged(exelist + ['-v'], msg='Detecting compiler via')
    except OSError:
        raise EnvironmentException('Could not execute Swift compiler: {}'.format(join_args(exelist)))
    version = search_version(err)
    if 'Swift' in err:
        with tempfile.NamedTemporaryFile(suffix='.swift') as f:
            cls = SwiftCompiler
            linker = guess_nix_linker(env, exelist, cls, version, for_machine, extra_args=[f.name])
        return cls(exelist, version, for_machine, is_cross, info, linker=linker)
    raise EnvironmentException('Unknown compiler: ' + join_args(exelist))