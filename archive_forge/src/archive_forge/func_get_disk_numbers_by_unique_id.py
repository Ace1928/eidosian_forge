import ctypes
import os
import re
import threading
from collections.abc import Iterable
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import pathutils
from os_win.utils import win32utils
from os_win.utils.winapi import libs as w_lib
def get_disk_numbers_by_unique_id(self, unique_id, unique_id_format):
    disks = self._get_disks_by_unique_id(unique_id, unique_id_format)
    return [disk.Number for disk in disks]