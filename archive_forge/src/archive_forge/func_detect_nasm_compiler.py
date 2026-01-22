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
def detect_nasm_compiler(env: 'Environment', for_machine: MachineChoice) -> Compiler:
    from .asm import NasmCompiler, YasmCompiler, MetrowerksAsmCompilerARM, MetrowerksAsmCompilerEmbeddedPowerPC
    compilers, _, _ = _get_compilers(env, 'nasm', for_machine)
    is_cross = env.is_cross_build(for_machine)
    cc = detect_c_compiler(env, for_machine)
    if not is_cross:
        from ..environment import detect_machine_info
        info = detect_machine_info({'c': cc})
    else:
        info = env.machines[for_machine]
    popen_exceptions: T.Dict[str, Exception] = {}
    for comp in compilers:
        if comp == ['nasm'] and is_windows() and (not shutil.which(comp[0])):
            default_path = os.path.join(os.environ['ProgramFiles'], 'NASM')
            comp[0] = shutil.which(comp[0], path=default_path) or comp[0]
        try:
            output = Popen_safe_logged(comp + ['--version'], msg='Detecting compiler via')[1]
        except OSError as e:
            popen_exceptions[' '.join(comp + ['--version'])] = e
            continue
        version = search_version(output)
        if 'NASM' in output:
            comp_class = NasmCompiler
            env.coredata.add_lang_args(comp_class.language, comp_class, for_machine, env)
            return comp_class([], comp, version, for_machine, info, cc.linker, is_cross=is_cross)
        elif 'yasm' in output:
            comp_class = YasmCompiler
            env.coredata.add_lang_args(comp_class.language, comp_class, for_machine, env)
            return comp_class([], comp, version, for_machine, info, cc.linker, is_cross=is_cross)
        elif 'Metrowerks' in output or 'Freescale' in output:
            if 'ARM' in output:
                comp_class_mwasmarm = MetrowerksAsmCompilerARM
                env.coredata.add_lang_args(comp_class_mwasmarm.language, comp_class_mwasmarm, for_machine, env)
                return comp_class_mwasmarm([], comp, version, for_machine, info, cc.linker, is_cross=is_cross)
            else:
                comp_class_mwasmeppc = MetrowerksAsmCompilerEmbeddedPowerPC
                env.coredata.add_lang_args(comp_class_mwasmeppc.language, comp_class_mwasmeppc, for_machine, env)
                return comp_class_mwasmeppc([], comp, version, for_machine, info, cc.linker, is_cross=is_cross)
    _handle_exceptions(popen_exceptions, compilers)
    raise EnvironmentException('Unreachable code (exception to make mypy happy)')