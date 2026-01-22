import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log
def find_executable(exe, path=None, _cache={}):
    """Return full path of a executable or None.

    Symbolic links are not followed.
    """
    key = (exe, path)
    try:
        return _cache[key]
    except KeyError:
        pass
    log.debug('find_executable(%r)' % exe)
    orig_exe = exe
    if path is None:
        path = os.environ.get('PATH', os.defpath)
    if os.name == 'posix':
        realpath = os.path.realpath
    else:
        realpath = lambda a: a
    if exe.startswith('"'):
        exe = exe[1:-1]
    suffixes = ['']
    if os.name in ['nt', 'dos', 'os2']:
        fn, ext = os.path.splitext(exe)
        extra_suffixes = ['.exe', '.com', '.bat']
        if ext.lower() not in extra_suffixes:
            suffixes = extra_suffixes
    if os.path.isabs(exe):
        paths = ['']
    else:
        paths = [os.path.abspath(p) for p in path.split(os.pathsep)]
    for path in paths:
        fn = os.path.join(path, exe)
        for s in suffixes:
            f_ext = fn + s
            if not os.path.islink(f_ext):
                f_ext = realpath(f_ext)
            if os.path.isfile(f_ext) and os.access(f_ext, os.X_OK):
                log.info('Found executable %s' % f_ext)
                _cache[key] = f_ext
                return f_ext
    log.warn('Could not locate executable %s' % orig_exe)
    return None