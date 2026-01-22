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
def detect_java_compiler(env: 'Environment', for_machine: MachineChoice) -> Compiler:
    from .java import JavaCompiler
    exelist = env.lookup_binary_entry(for_machine, 'java')
    info = env.machines[for_machine]
    if exelist is None:
        exelist = [defaults['java'][0]]
    try:
        p, out, err = Popen_safe_logged(exelist + ['-version'], msg='Detecting compiler via')
    except OSError:
        raise EnvironmentException('Could not execute Java compiler: {}'.format(join_args(exelist)))
    if 'javac' in out or 'javac' in err:
        version = search_version(err if 'javac' in err else out)
        if not version or version == 'unknown version':
            parts = (err if 'javac' in err else out).split()
            if len(parts) > 1:
                version = parts[1]
        comp_class = JavaCompiler
        env.coredata.add_lang_args(comp_class.language, comp_class, for_machine, env)
        return comp_class(exelist, version, for_machine, info)
    raise EnvironmentException('Unknown compiler: ' + join_args(exelist))