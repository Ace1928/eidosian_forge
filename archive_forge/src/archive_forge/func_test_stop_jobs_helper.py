from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
@ddt.data(True, False)
@mock.patch.object(jobutils.JobUtils, '_get_pending_jobs_affecting_element')
def test_stop_jobs_helper(self, jobs_ended, mock_get_pending_jobs):
    mock_job1 = mock.Mock(Cancellable=True)
    mock_job2 = mock.Mock(Cancellable=True)
    mock_job3 = mock.Mock(Cancellable=False)
    pending_jobs = [mock_job1, mock_job2, mock_job3]
    mock_get_pending_jobs.side_effect = (pending_jobs, pending_jobs if not jobs_ended else [])
    mock_job1.RequestStateChange.side_effect = test_base.FakeWMIExc(hresult=jobutils._utils._WBEM_E_NOT_FOUND)
    mock_job2.RequestStateChange.side_effect = test_base.FakeWMIExc(hresult=mock.sentinel.hresult)
    if jobs_ended:
        self.jobutils._stop_jobs(mock.sentinel.vm)
    else:
        self.assertRaises(exceptions.JobTerminateFailed, self.jobutils._stop_jobs, mock.sentinel.vm)
    mock_get_pending_jobs.assert_has_calls([mock.call(mock.sentinel.vm)] * 2)
    mock_job1.RequestStateChange.assert_called_once_with(self.jobutils._KILL_JOB_STATE_CHANGE_REQUEST)
    mock_job2.RequestStateChange.assert_called_once_with(self.jobutils._KILL_JOB_STATE_CHANGE_REQUEST)
    self.assertFalse(mock_job3.RequestStateqqChange.called)