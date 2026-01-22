import ctypes
from oslo_log import log as logging
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import kernel32 as kernel32_struct
def create_job_object(self, name=None):
    """Create or open a job object.

        :param name: (Optional) the job name.
        :returns: a handle of the created job.
        """
    pname = None if name is None else ctypes.c_wchar_p(name)
    return self._run_and_check_output(kernel32.CreateJobObjectW, None, pname, error_ret_vals=[None])