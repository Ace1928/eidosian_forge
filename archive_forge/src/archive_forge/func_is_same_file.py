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
def is_same_file(self, path_a, path_b):
    """Check if two paths point to the same file."""
    file_a_id = self.get_file_id(path_a)
    file_b_id = self.get_file_id(path_b)
    return file_a_id == file_b_id