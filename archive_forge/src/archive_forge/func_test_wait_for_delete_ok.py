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
def test_wait_for_delete_ok(self, mock_sleep):
    resource = mock.MagicMock(status='ACTIVE', progress=None)
    mock_get = mock.Mock(side_effect=[resource, exceptions.NotFound(404)])
    manager = mock.MagicMock(get=mock_get)
    res_id = str(uuid.uuid4())
    callback = mock.Mock()
    self.assertTrue(utils.wait_for_delete(manager, res_id, callback=callback))
    mock_sleep.assert_called_once_with(5)
    callback.assert_called_once_with(0)