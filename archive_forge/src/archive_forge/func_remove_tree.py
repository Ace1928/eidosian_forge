import os
import errno
from distutils.errors import DistutilsFileError, DistutilsInternalError
from distutils import log
def remove_tree(directory, verbose=1, dry_run=0):
    """Recursively remove an entire directory tree.

    Any errors are ignored (apart from being reported to stdout if 'verbose'
    is true).
    """
    global _path_created
    if verbose >= 1:
        log.info("removing '%s' (and everything under it)", directory)
    if dry_run:
        return
    cmdtuples = []
    _build_cmdtuple(directory, cmdtuples)
    for cmd in cmdtuples:
        try:
            cmd[0](cmd[1])
            abspath = os.path.abspath(cmd[1])
            if abspath in _path_created:
                del _path_created[abspath]
        except OSError as exc:
            log.warn('error removing %s: %s', directory, exc)