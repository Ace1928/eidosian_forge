from __future__ import annotations
import os
import sys
import argparse
import pickle
import subprocess
import typing as T
import locale
from ..utils.core import ExecutableSerialisation
def run_exe(exe: ExecutableSerialisation, extra_env: T.Optional[T.Dict[str, str]]=None) -> int:
    if exe.exe_wrapper:
        if not exe.exe_wrapper.found():
            raise AssertionError("BUG: Can't run cross-compiled exe {!r} with not-found wrapper {!r}".format(exe.cmd_args[0], exe.exe_wrapper.get_path()))
        cmd_args = exe.exe_wrapper.get_command() + exe.cmd_args
    else:
        cmd_args = exe.cmd_args
    child_env = os.environ.copy()
    if extra_env:
        child_env.update(extra_env)
    if exe.env:
        child_env = exe.env.get_env(child_env)
    if exe.extra_paths:
        child_env['PATH'] = os.pathsep.join(exe.extra_paths + ['']) + child_env['PATH']
        if exe.exe_wrapper and any(('wine' in i for i in exe.exe_wrapper.get_command())):
            from .. import mesonlib
            child_env['WINEPATH'] = mesonlib.get_wine_shortpath(exe.exe_wrapper.get_command(), ['Z:' + p for p in exe.extra_paths] + child_env.get('WINEPATH', '').split(';'), exe.workdir)
    stdin = None
    if exe.feed:
        stdin = open(exe.feed, 'rb')
    pipe = subprocess.PIPE
    if exe.verbose:
        assert not exe.capture, 'Cannot capture and print to console at the same time'
        pipe = None
    p = subprocess.Popen(cmd_args, env=child_env, cwd=exe.workdir, close_fds=False, stdin=stdin, stdout=pipe, stderr=pipe)
    stdout, stderr = p.communicate()
    if stdin is not None:
        stdin.close()
    if p.returncode == 3221225781:
        strerror = 'Failed to run due to missing DLLs, with path: ' + child_env['PATH']
        raise FileNotFoundError(p.returncode, strerror, cmd_args)
    if p.returncode != 0:
        if exe.pickled:
            print(f'while executing {cmd_args!r}')
        if exe.verbose:
            return p.returncode
        encoding = locale.getpreferredencoding()
        if not exe.capture:
            print('--- stdout ---')
            print(stdout.decode(encoding=encoding, errors='replace'))
        print('--- stderr ---')
        print(stderr.decode(encoding=encoding, errors='replace'))
        return p.returncode
    if exe.capture:
        skip_write = False
        try:
            with open(exe.capture, 'rb') as cur:
                skip_write = cur.read() == stdout
        except OSError:
            pass
        if not skip_write:
            with open(exe.capture, 'wb') as output:
                output.write(stdout)
    return 0