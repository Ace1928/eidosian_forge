import ctypes
from oslo_log import log as logging
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import kernel32 as kernel32_struct
Associates a new job to the specified process.

        The process is immediately killed when the last job handle is closed.
        This mechanism can be useful when ensuring that child processes get
        killed along with a parent process.

        This method does not check if the specified process is already part of
        a job. Starting with WS 2012, nested jobs are available.

        :returns: the job handle, if a job was successfully created and
                  associated with the process, otherwise "None".
        