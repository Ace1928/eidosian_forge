from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
@mock.patch.object(jobutils._utils, '_is_not_found_exc')
def test_get_pending_jobs_ignored(self, mock_is_not_found_exc):
    mock_not_found_mapping = mock.MagicMock()
    type(mock_not_found_mapping).AffectingElement = mock.PropertyMock(side_effect=exceptions.x_wmi)
    self.jobutils._conn.Msvm_AffectedJobElement.return_value = [mock_not_found_mapping]
    pending_jobs = self.jobutils._get_pending_jobs_affecting_element(mock.MagicMock())
    self.assertEqual([], pending_jobs)