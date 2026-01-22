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
@contextlib.contextmanager
def temporary_file(self, suffix=None, *args, **kwargs):
    """Creates a random, temporary, closed file, returning the file's

        path. It's different from tempfile.NamedTemporaryFile which returns
        an open file descriptor.
        """
    tmp_file_path = None
    try:
        tmp_file_path = self.create_temporary_file(suffix, *args, **kwargs)
        yield tmp_file_path
    finally:
        if tmp_file_path:
            fileutils.delete_if_exists(tmp_file_path)