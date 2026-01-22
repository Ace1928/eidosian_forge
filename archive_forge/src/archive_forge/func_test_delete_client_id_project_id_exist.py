from unittest import mock
import uuid
from openstack.message.v2 import queue
from openstack.tests.unit import base
def test_delete_client_id_project_id_exist(self):
    sess = mock.Mock()
    resp = mock.Mock()
    sess.delete.return_value = resp
    sot = queue.Queue(**FAKE2)
    sot._translate_response = mock.Mock()
    sot.delete(sess)
    url = 'queues/%s' % FAKE2['name']
    headers = {'Client-ID': 'OLD_CLIENT_ID', 'X-PROJECT-ID': 'OLD_PROJECT_ID'}
    sess.delete.assert_called_with(url, headers=headers)
    sot._translate_response.assert_called_once_with(resp, has_body=False)