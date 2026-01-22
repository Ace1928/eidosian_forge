import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def get_home_dir(require_writable=False) -> str:
    """Return the 'home' directory, as a unicode string.

    Uses os.path.expanduser('~'), and checks for writability.

    See stdlib docs for how this is determined.
    For Python <3.8, $HOME is first priority on *ALL* platforms.
    For Python >=3.8 on Windows, %HOME% is no longer considered.

    Parameters
    ----------
    require_writable : bool [default: False]
        if True:
            guarantees the return value is a writable directory, otherwise
            raises HomeDirError
        if False:
            The path is resolved, but it is not guaranteed to exist or be writable.
    """
    homedir = os.path.expanduser('~')
    homedir = os.path.realpath(homedir)
    if not _writable_dir(homedir) and os.name == 'nt':
        try:
            import winreg as wreg
            with wreg.OpenKey(wreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders') as key:
                homedir = wreg.QueryValueEx(key, 'Personal')[0]
        except:
            pass
    if not require_writable or _writable_dir(homedir):
        assert isinstance(homedir, str), 'Homedir should be unicode not bytes'
        return homedir
    else:
        raise HomeDirError('%s is not a writable dir, set $HOME environment variable to override' % homedir)