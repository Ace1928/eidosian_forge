import io
import json
import os
import platform
import shutil
import subprocess
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from cmdstanpy.utils import get_logger
from cmdstanpy.utils.cmdstan import (
from cmdstanpy.utils.command import do_command
from cmdstanpy.utils.filesystem import SanitizedOrTmpFilePath
def compile_stan_file(src: Union[str, Path], force: bool=False, stanc_options: Optional[Dict[str, Any]]=None, cpp_options: Optional[Dict[str, Any]]=None, user_header: OptionalPath=None) -> str:
    """
    Compile the given Stan program file.  Translates the Stan code to
    C++, then calls the C++ compiler.

    By default, this function compares the timestamps on the source and
    executable files; if the executable is newer than the source file, it
    will not recompile the file, unless argument ``force`` is ``True``
    or unless the compiler options have been changed.

    :param src: Path to Stan program file.

    :param force: When ``True``, always compile, even if the executable file
        is newer than the source file.  Used for Stan models which have
        ``#include`` directives in order to force recompilation when changes
        are made to the included files.

    :param stanc_options: Options for stanc compiler.
    :param cpp_options: Options for C++ compiler.
    :param user_header: A path to a header file to include during C++
        compilation.
    """
    src = Path(src).resolve()
    if not src.exists():
        raise ValueError(f'stan file does not exist: {src}')
    compiler_options = CompilerOptions(stanc_options=stanc_options, cpp_options=cpp_options, user_header=user_header)
    compiler_options.validate()
    exe_target = src.with_suffix(EXTENSION)
    if exe_target.exists():
        exe_time = os.path.getmtime(exe_target)
        included_files = [src]
        included_files.extend(src_info(str(src), compiler_options).get('included_files', []))
        out_of_date = any((os.path.getmtime(included_file) > exe_time for included_file in included_files))
        if not out_of_date and (not force):
            get_logger().debug('found newer exe file, not recompiling')
            return str(exe_target)
    compilation_failed = False
    with SanitizedOrTmpFilePath(str(src)) as (stan_file, is_copied):
        exe_file = os.path.splitext(stan_file)[0] + EXTENSION
        hpp_file = os.path.splitext(exe_file)[0] + '.hpp'
        if os.path.exists(hpp_file):
            os.remove(hpp_file)
        if os.path.exists(exe_file):
            get_logger().debug('Removing %s', exe_file)
            os.remove(exe_file)
        get_logger().info('compiling stan file %s to exe file %s', stan_file, exe_target)
        make = os.getenv('MAKE', 'make' if platform.system() != 'Windows' else 'mingw32-make')
        cmd = [make]
        cmd.extend(compiler_options.compose(filename_in_msg=src.name))
        cmd.append(Path(exe_file).as_posix())
        sout = io.StringIO()
        try:
            do_command(cmd=cmd, cwd=cmdstan_path(), fd_out=sout)
        except RuntimeError as e:
            sout.write(f'\n{str(e)}\n')
            compilation_failed = True
        finally:
            console = sout.getvalue()
        get_logger().debug('Console output:\n%s', console)
        if not compilation_failed:
            if is_copied:
                shutil.copy(exe_file, exe_target)
            get_logger().info('compiled model executable: %s', exe_target)
        if 'Warning' in console:
            lines = console.split('\n')
            warnings = [x for x in lines if x.startswith('Warning')]
            get_logger().warning('Stan compiler has produced %d warnings:', len(warnings))
            get_logger().warning(console)
        if compilation_failed:
            if 'PCH' in console or 'precompiled header' in console:
                get_logger().warning("CmdStan's precompiled header (PCH) files may need to be rebuilt.Please run cmdstanpy.rebuild_cmdstan().\nIf the issue persists please open a bug report")
            raise ValueError(f"Failed to compile Stan model '{src}'. Console:\n{console}")
        return str(exe_target)