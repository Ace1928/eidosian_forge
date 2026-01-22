import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def cleanup_tmpdir(tmpdir=None, keep_so=False):
    """Clean up the temporary directory by removing all files in it
    called `_cffi_*.{c,so}` as well as the `build` subdirectory."""
    tmpdir = tmpdir or _caller_dir_pycache()
    try:
        filelist = os.listdir(tmpdir)
    except OSError:
        return
    if keep_so:
        suffix = '.c'
    else:
        suffix = _get_so_suffixes()[0].lower()
    for fn in filelist:
        if fn.lower().startswith('_cffi_') and (fn.lower().endswith(suffix) or fn.lower().endswith('.c')):
            try:
                os.unlink(os.path.join(tmpdir, fn))
            except OSError:
                pass
    clean_dir = [os.path.join(tmpdir, 'build')]
    for dir in clean_dir:
        try:
            for fn in os.listdir(dir):
                fn = os.path.join(dir, fn)
                if os.path.isdir(fn):
                    clean_dir.append(fn)
                else:
                    os.unlink(fn)
        except OSError:
            pass