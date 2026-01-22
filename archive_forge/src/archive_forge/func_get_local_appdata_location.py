import glob
import os
import struct
from .lazy_import import lazy_import
import ctypes
from breezy import cmdline
from breezy.i18n import gettext
def get_local_appdata_location():
    """Return Local Application Data location.
    Return the same as get_appdata_location() if we cannot obtain location.

    Windows defines two 'Application Data' folders per user - a 'roaming'
    one that moves with the user as they logon to different machines, and
    a 'local' one that stays local to the machine.  This returns the 'local'
    directory, and thus is suitable for caches, temp files and other things
    which don't need to move with the user.
    """
    local = _get_sh_special_folder_path(CSIDL_LOCAL_APPDATA)
    if local:
        return local
    local = os.environ.get('LOCALAPPDATA')
    if local:
        return local
    return get_appdata_location()