from unittest import mock
import testtools
from heatclient.common import hook_utils
import heatclient.v1.shell as shell
def test_clear_pre_create_hooks(self):
    type(self.args).hook = mock.PropertyMock(return_value=['bp'])
    type(self.args).pre_create = mock.PropertyMock(return_value=True)
    bp = mock.Mock()
    type(bp).resource_name = 'bp'
    self.client.resources.list = mock.Mock(return_value=[bp])
    shell.do_hook_clear(self.client, self.args)
    self.assertEqual(1, self.client.resources.signal.call_count)
    payload = self.client.resources.signal.call_args_list[0][1]
    self.assertEqual({'unset_hook': 'pre-create'}, payload['data'])
    self.assertEqual('bp', payload['resource_name'])
    self.assertEqual('mystack', payload['stack_id'])