import atexit
import errno
import math
import numbers
import os
import platform as _platform
import signal as _signal
import sys
import warnings
from contextlib import contextmanager
from billiard.compat import close_open_fds, get_fdmax
from billiard.util import set_pdeathsig as _set_pdeathsig
from kombu.utils.compat import maybe_fileno
from kombu.utils.encoding import safe_str
from .exceptions import SecurityError, SecurityWarning, reraise
from .local import try_import
def check_privileges(accept_content):
    if grp is None or pwd is None:
        return
    pickle_or_serialize = 'pickle' in accept_content or 'application/group-python-serialize' in accept_content
    uid = os.getuid() if hasattr(os, 'getuid') else 65535
    gid = os.getgid() if hasattr(os, 'getgid') else 65535
    euid = os.geteuid() if hasattr(os, 'geteuid') else 65535
    egid = os.getegid() if hasattr(os, 'getegid') else 65535
    if hasattr(os, 'fchown'):
        if not all((hasattr(os, attr) for attr in ('getuid', 'getgid', 'geteuid', 'getegid'))):
            raise SecurityError('suspicious platform, contact support')
    try:
        gid_entry = grp.getgrgid(gid)
        egid_entry = grp.getgrgid(egid)
    except KeyError:
        warnings.warn(SecurityWarning(ASSUMING_ROOT))
        _warn_or_raise_security_error(egid, euid, gid, uid, pickle_or_serialize)
        return
    gid_grp_name = gid_entry[0]
    egid_grp_name = egid_entry[0]
    gids_in_use = (gid_grp_name, egid_grp_name)
    groups_with_security_risk = ('sudo', 'wheel')
    is_root = uid == 0 or euid == 0
    if is_root or any((group in gids_in_use for group in groups_with_security_risk)):
        _warn_or_raise_security_error(egid, euid, gid, uid, pickle_or_serialize)