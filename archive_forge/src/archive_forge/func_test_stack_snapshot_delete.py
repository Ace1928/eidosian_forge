from tempest.lib.common.utils import data_utils as utils
from heatclient.tests.functional import config
from heatclient.tests.functional.osc.v1 import base
def test_stack_snapshot_delete(self):
    snapshot_name = utils.rand_name(name='test-stack-snapshot')
    stack = self._stack_create_minimal()
    snapshot = self._stack_snapshot_create(stack['id'], snapshot_name)
    self._stack_snapshot_delete(stack['id'], snapshot['id'])
    stacks_raw = self.openstack('stack snapshot list' + ' ' + self.stack_name)
    self.assertNotIn(snapshot['id'], stacks_raw)