import contextlib
import ctypes
import os
import shutil
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils import _acl_utils
from os_win.utils.io import ioutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
def move_folder_files(self, src_dir, dest_dir):
    """Moves the files of the given src_dir to dest_dir.

        It will ignore any nested folders.

        :param src_dir: Given folder from which to move files.
        :param dest_dir: Folder to which to move files.
        """
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            self.rename(src, os.path.join(dest_dir, fname))