from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_module_apply(self):
    resp = mock.Mock()
    resp.status_code = 200
    body = {'modules': []}
    self.instances.api.client.post = mock.Mock(return_value=(resp, body))
    self.instances.module_apply(self.instance_with_id, 'mod_id')
    resp.status_code = 500
    self.assertRaises(Exception, self.instances.module_apply, 'instance1', 'mod1')