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
def detect_d_compiler(env: 'Environment', for_machine: MachineChoice) -> Compiler:
    from . import c, d
    info = env.machines[for_machine]
    c_compiler = {'c': detect_c_compiler(env, for_machine)}
    is_msvc = isinstance(c_compiler['c'], c.VisualStudioCCompiler)
    if not is_msvc:
        c_compiler = {}
    from ..environment import detect_cpu_family
    arch = detect_cpu_family(c_compiler)
    if is_msvc and arch == 'x86':
        arch = 'x86_mscoff'
    popen_exceptions = {}
    is_cross = env.is_cross_build(for_machine)
    compilers, ccache, exe_wrap = _get_compilers(env, 'd', for_machine)
    cls: T.Type[d.DCompiler]
    for exelist in compilers:
        if os.path.basename(exelist[-1]).startswith(('ldmd', 'gdmd')):
            raise EnvironmentException(f'Meson does not support {exelist[-1]} as it is only a DMD frontend for another compiler.Please provide a valid value for DC or unset it so that Meson can resolve the compiler by itself.')
        try:
            p, out = Popen_safe(exelist + ['--version'])[0:2]
        except OSError as e:
            popen_exceptions[join_args(exelist + ['--version'])] = e
            continue
        version = search_version(out)
        full_version = out.split('\n', 1)[0]
        if 'LLVM D compiler' in out:
            cls = d.LLVMDCompiler
            o, f = tempfile.mkstemp('.d')
            os.close(o)
            try:
                if info.is_windows() or info.is_cygwin():
                    objfile = os.path.basename(f)[:-1] + 'obj'
                    linker = guess_win_linker(env, exelist, cls, full_version, for_machine, use_linker_prefix=True, invoked_directly=False, extra_args=[f])
                else:
                    objfile = os.path.basename(f)[:-1] + 'o'
                    linker = guess_nix_linker(env, exelist, cls, full_version, for_machine, extra_args=[f])
            finally:
                windows_proof_rm(f)
                windows_proof_rm(objfile)
            return cls(exelist, version, for_machine, info, arch, full_version=full_version, linker=linker, version_output=out)
        elif 'gdc' in out:
            cls = d.GnuDCompiler
            linker = guess_nix_linker(env, exelist, cls, version, for_machine)
            return cls(exelist, version, for_machine, info, arch, exe_wrapper=exe_wrap, is_cross=is_cross, full_version=full_version, linker=linker)
        elif 'The D Language Foundation' in out or 'Digital Mars' in out:
            cls = d.DmdDCompiler
            o, f = tempfile.mkstemp('.d')
            os.close(o)
            arch_arg = '-m64' if arch == 'x86_64' else '-m32'
            try:
                if info.is_windows() or info.is_cygwin():
                    objfile = os.path.basename(f)[:-1] + 'obj'
                    linker = guess_win_linker(env, exelist, cls, full_version, for_machine, invoked_directly=False, extra_args=[f, arch_arg])
                else:
                    objfile = os.path.basename(f)[:-1] + 'o'
                    linker = guess_nix_linker(env, exelist, cls, full_version, for_machine, extra_args=[f, arch_arg])
            finally:
                windows_proof_rm(f)
                windows_proof_rm(objfile)
            return cls(exelist, version, for_machine, info, arch, full_version=full_version, linker=linker)
        raise EnvironmentException('Unknown compiler: ' + join_args(exelist))
    _handle_exceptions(popen_exceptions, compilers)
    raise EnvironmentException('Unreachable code (exception to make mypy happy)')