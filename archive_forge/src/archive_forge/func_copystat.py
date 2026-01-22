import os
import sys
import stat
import fnmatch
import collections
import errno
def copystat(src, dst, *, follow_symlinks=True):
    """Copy file metadata

    Copy the permission bits, last access time, last modification time, and
    flags from `src` to `dst`. On Linux, copystat() also copies the "extended
    attributes" where possible. The file contents, owner, and group are
    unaffected. `src` and `dst` are path-like objects or path names given as
    strings.

    If the optional flag `follow_symlinks` is not set, symlinks aren't
    followed if and only if both `src` and `dst` are symlinks.
    """
    sys.audit('shutil.copystat', src, dst)

    def _nop(*args, ns=None, follow_symlinks=None):
        pass
    follow = follow_symlinks or not (_islink(src) and os.path.islink(dst))
    if follow:

        def lookup(name):
            return getattr(os, name, _nop)
    else:

        def lookup(name):
            fn = getattr(os, name, _nop)
            if fn in os.supports_follow_symlinks:
                return fn
            return _nop
    if isinstance(src, os.DirEntry):
        st = src.stat(follow_symlinks=follow)
    else:
        st = lookup('stat')(src, follow_symlinks=follow)
    mode = stat.S_IMODE(st.st_mode)
    lookup('utime')(dst, ns=(st.st_atime_ns, st.st_mtime_ns), follow_symlinks=follow)
    _copyxattr(src, dst, follow_symlinks=follow)
    _chmod = lookup('chmod')
    if os.name == 'nt':
        if follow:
            if os.path.islink(dst):
                dst = os.path.realpath(dst, strict=True)
        else:

            def _chmod(*args, **kwargs):
                os.chmod(*args)
    try:
        _chmod(dst, mode, follow_symlinks=follow)
    except NotImplementedError:
        pass
    if hasattr(st, 'st_flags'):
        try:
            lookup('chflags')(dst, st.st_flags, follow_symlinks=follow)
        except OSError as why:
            for err in ('EOPNOTSUPP', 'ENOTSUP'):
                if hasattr(errno, err) and why.errno == getattr(errno, err):
                    break
            else:
                raise