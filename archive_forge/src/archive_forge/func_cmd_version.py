import json
import os
import sys
import traceback
from _pydev_bundle import pydev_log
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydevd_bundle import pydevd_traceproperty, pydevd_dont_trace, pydevd_utils
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_breakpoints import get_exception_class
from _pydevd_bundle.pydevd_comm import (
from _pydevd_bundle.pydevd_constants import NEXT_VALUE_SEPARATOR, IS_WINDOWS, NULL
from _pydevd_bundle.pydevd_comm_constants import ID_TO_MEANING, CMD_EXEC_EXPRESSION, CMD_AUTHENTICATE
from _pydevd_bundle.pydevd_api import PyDevdAPI
from io import StringIO
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id
import pydevd_file_utils
def cmd_version(self, py_db, cmd_id, seq, text):
    if IS_WINDOWS:
        ide_os = 'WINDOWS'
    else:
        ide_os = 'UNIX'
    breakpoints_by = 'LINE'
    splitted = text.split('\t')
    if len(splitted) == 1:
        _local_version = splitted
    elif len(splitted) == 2:
        _local_version, ide_os = splitted
    elif len(splitted) == 3:
        _local_version, ide_os, breakpoints_by = splitted
    version_msg = self.api.set_ide_os_and_breakpoints_by(py_db, seq, ide_os, breakpoints_by)
    self.api.set_enable_thread_notifications(py_db, True)
    return version_msg