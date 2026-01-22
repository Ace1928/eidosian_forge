import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def getsitepackages(prefixes=None):
    """Returns a list containing all global site-packages directories.

    For each directory present in ``prefixes`` (or the global ``PREFIXES``),
    this function will find its `site-packages` subdirectory depending on the
    system environment, and will return a list of full paths.
    """
    sitepackages = []
    seen = set()
    if prefixes is None:
        prefixes = PREFIXES
    for prefix in prefixes:
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        if os.sep == '/':
            libdirs = [sys.platlibdir]
            if sys.platlibdir != 'lib':
                libdirs.append('lib')
            for libdir in libdirs:
                path = os.path.join(prefix, libdir, 'python%d.%d' % sys.version_info[:2], 'site-packages')
                sitepackages.append(path)
        else:
            sitepackages.append(prefix)
            sitepackages.append(os.path.join(prefix, 'Lib', 'site-packages'))
    return sitepackages