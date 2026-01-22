import os
import time
import yaml
from . import config, debug, errors, lock, osutils, ui, urlutils
from .decorators import only_raises
from .errors import (DirectoryNotEmpty, LockBreakMismatch, LockBroken,
from .i18n import gettext
from .osutils import format_delta, get_host_name, rand_chars
from .trace import mutter, note
from .transport import FileExists, NoSuchFile
def is_lock_holder_known_dead(self):
    """True if the lock holder process is known to be dead.

        False if it's either known to be still alive, or if we just can't tell.

        We can be fairly sure the lock holder is dead if it declared the same
        hostname and there is no process with the given pid alive.  If people
        have multiple machines with the same hostname this may cause trouble.

        This doesn't check whether the lock holder is in fact the same process
        calling this method.  (In that case it will return true.)
        """
    if self.get('hostname') != get_host_name():
        return False
    if self.get('hostname') == 'localhost':
        return False
    if self.get('user') != get_username_for_lock_info():
        return False
    pid_str = self.info_dict.get('pid', None)
    if not pid_str:
        mutter('no pid recorded in {!r}'.format(self))
        return False
    try:
        pid = int(pid_str)
    except ValueError:
        mutter("can't parse pid %r from %r" % (pid_str, self))
        return False
    return osutils.is_local_pid_dead(pid)