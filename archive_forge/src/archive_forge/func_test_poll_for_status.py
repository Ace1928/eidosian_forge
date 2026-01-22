from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@mock.patch('cinderclient.shell_utils.time')
def test_poll_for_status(self, mock_time):
    poll_period = 2
    some_id = 'some-id'
    global_request_id = 'req-someid'
    action = 'some'
    updated_objects = (base.Resource(None, info={'not_default_field': 'creating'}), base.Resource(None, info={'not_default_field': 'available'}))
    poll_fn = mock.MagicMock(side_effect=updated_objects)
    cinderclient.shell_utils._poll_for_status(poll_fn=poll_fn, obj_id=some_id, global_request_id=global_request_id, messages=base.Resource(None, {}), info={}, action=action, status_field='not_default_field', final_ok_states=['available'], timeout_period=3600)
    self.assertEqual([mock.call(poll_period)] * 2, mock_time.sleep.call_args_list)
    self.assertEqual([mock.call(some_id)] * 2, poll_fn.call_args_list)