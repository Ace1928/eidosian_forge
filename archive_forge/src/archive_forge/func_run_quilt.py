import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
def run_quilt(args, working_dir, series_file=None, patches_dir=None, quiet=None):
    """Run quilt.

    :param args: Arguments to quilt
    :param working_dir: Working dir
    :param series_file: Optional path to the series file
    :param patches_dir: Optional path to the patches
    :param quilt: Whether to be quiet (quilt stderr not to terminal)
    :raise QuiltError: When running quilt fails
    """

    def subprocess_setup():
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    env = {}
    if patches_dir is not None:
        env['QUILT_PATCHES'] = patches_dir
    else:
        env['QUILT_PATCHES'] = os.path.join(working_dir, DEFAULT_PATCHES_DIR)
    if series_file is not None:
        env['QUILT_SERIES'] = series_file
    else:
        env['QUILT_SERIES'] = DEFAULT_SERIES_FILE
    if quiet is None:
        quiet = trace.is_quiet()
    if not quiet:
        stderr = subprocess.STDOUT
    else:
        stderr = subprocess.PIPE
    quilt_path = osutils.find_executable_on_path('quilt')
    if quilt_path is None:
        raise QuiltNotInstalled()
    command = [quilt_path] + args
    trace.mutter('running: %r', command)
    if not os.path.isdir(working_dir):
        raise AssertionError('%s is not a valid directory' % working_dir)
    try:
        proc = subprocess.Popen(command, cwd=working_dir, env=env, stdin=subprocess.PIPE, preexec_fn=subprocess_setup, stdout=subprocess.PIPE, stderr=stderr)
    except FileNotFoundError as e:
        raise QuiltNotInstalled() from e
    stdout, stderr = proc.communicate()
    if proc.returncode not in (0, 2):
        if stdout is not None:
            stdout = stdout.decode()
        if stderr is not None:
            stderr = stderr.decode()
        raise QuiltError(proc.returncode, stdout, stderr)
    if stdout is None:
        return ''
    return stdout