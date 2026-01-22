from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_target_list_update_delete(self):
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    self.create_volume_target(boot_index=0, volume_id='04452bed-5367-4202-8bf5-de4335ac57h3', node_id=self.node.id, volume_type='iscsi', extra={'foo': 'bar'})
    volume_target = next(self.conn.baremetal.volume_targets(details=True, node=self.node.id))
    self.assertEqual(volume_target.extra, {'foo': 'bar'})
    self.conn.baremetal.update_volume_target(volume_target, extra={'foo': 42})
    self.conn.baremetal.delete_volume_target(volume_target, ignore_missing=False)