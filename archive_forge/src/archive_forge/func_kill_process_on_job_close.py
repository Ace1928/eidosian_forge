import ctypes
from oslo_log import log as logging
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import kernel32 as kernel32_struct
def kill_process_on_job_close(self, pid):
    """Associates a new job to the specified process.

        The process is immediately killed when the last job handle is closed.
        This mechanism can be useful when ensuring that child processes get
        killed along with a parent process.

        This method does not check if the specified process is already part of
        a job. Starting with WS 2012, nested jobs are available.

        :returns: the job handle, if a job was successfully created and
                  associated with the process, otherwise "None".
        """
    process_handle = None
    job_handle = None
    job_associated = False
    try:
        desired_process_access = w_const.PROCESS_SET_QUOTA | w_const.PROCESS_TERMINATE
        process_handle = self.open_process(pid, desired_process_access)
        job_handle = self.create_job_object()
        job_info = kernel32_struct.JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        job_info.BasicLimitInformation.LimitFlags = w_const.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        job_info_class = w_const.JobObjectExtendedLimitInformation
        self.set_information_job_object(job_handle, job_info_class, job_info)
        self.assign_process_to_job_object(job_handle, process_handle)
        job_associated = True
    finally:
        if process_handle:
            self._win32_utils.close_handle(process_handle)
        if not job_associated and job_handle:
            self._win32_utils.close_handle(job_handle)
    return job_handle