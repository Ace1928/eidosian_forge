from unittest import mock
import uuid
from openstack.message.v2 import message
from openstack.tests.unit import base
@mock.patch.object(uuid, 'uuid4')
def test_delete_unclaimed(self, mock_uuid):
    sess = mock.Mock()
    resp = mock.Mock()
    sess.delete.return_value = resp
    sess.get_project_id.return_value = 'NEW_PROJECT_ID'
    mock_uuid.return_value = 'NEW_CLIENT_ID'
    sot = message.Message(**FAKE1)
    sot.claim_id = None
    sot._translate_response = mock.Mock()
    sot.delete(sess)
    url = 'queues/%(queue)s/messages/%(message)s' % {'queue': FAKE1['queue_name'], 'message': FAKE1['id']}
    headers = {'Client-ID': 'NEW_CLIENT_ID', 'X-PROJECT-ID': 'NEW_PROJECT_ID'}
    sess.delete.assert_called_with(url, headers=headers)
    sess.get_project_id.assert_called_once_with()
    sot._translate_response.assert_called_once_with(resp, has_body=False)