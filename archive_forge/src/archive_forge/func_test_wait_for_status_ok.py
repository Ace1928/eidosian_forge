import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
@mock.patch.object(time, 'sleep')
def test_wait_for_status_ok(self, mock_sleep):
    resource = mock.MagicMock(status='ACTIVE')
    status_f = mock.Mock(return_value=resource)
    res_id = str(uuid.uuid4())
    self.assertTrue(utils.wait_for_status(status_f, res_id))
    mock_sleep.assert_not_called()