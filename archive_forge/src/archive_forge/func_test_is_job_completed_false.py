from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_is_job_completed_false(self):
    job = mock.MagicMock(JobState=constants.WMI_JOB_STATE_RUNNING)
    self.assertFalse(self.jobutils._is_job_completed(job))