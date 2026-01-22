import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def symlink_or_copy(self, src, dst, relative_symlinks_ok=False):
    """
            Try symlinking a file, and if that fails, fall back to copying.
            """
    bad_src = os.path.lexists(src) and (not os.path.exists(src))
    if self.symlinks and (not bad_src) and (not os.path.islink(dst)):
        try:
            if relative_symlinks_ok:
                assert os.path.dirname(src) == os.path.dirname(dst)
                os.symlink(os.path.basename(src), dst)
            else:
                os.symlink(src, dst)
            return
        except Exception:
            logger.warning('Unable to symlink %r to %r', src, dst)
    basename, ext = os.path.splitext(os.path.basename(src))
    srcfn = os.path.join(os.path.dirname(__file__), 'scripts', 'nt', basename + ext)
    if sysconfig.is_python_build() or not os.path.isfile(srcfn):
        if basename.endswith('_d'):
            ext = '_d' + ext
            basename = basename[:-2]
        if basename == 'python':
            basename = 'venvlauncher'
        elif basename == 'pythonw':
            basename = 'venvwlauncher'
        src = os.path.join(os.path.dirname(src), basename + ext)
    else:
        src = srcfn
    if not os.path.exists(src):
        if not bad_src:
            logger.warning('Unable to copy %r', src)
        return
    shutil.copyfile(src, dst)