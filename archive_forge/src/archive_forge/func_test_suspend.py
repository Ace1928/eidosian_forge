from unittest import mock
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit import test_resource
def test_suspend(self):
    sess = mock.Mock()
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.headers = {}
    mock_response.json.return_value = {}
    sess.post = mock.Mock(return_value=mock_response)
    url = 'stacks/%s/actions' % FAKE_ID
    body = {'suspend': None}
    sot = stack.Stack(**FAKE)
    res = sot.suspend(sess)
    self.assertIsNone(res)
    sess.post.assert_called_with(url, json=body, microversion=None)