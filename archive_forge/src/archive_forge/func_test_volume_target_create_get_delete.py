from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_target_create_get_delete(self):
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    volume_target = self.create_volume_target(boot_index=0, volume_id='04452bed-5367-4202-8bf5-de4335ac56d2', volume_type='iscsi')
    loaded = self.conn.baremetal.get_volume_target(volume_target.id)
    self.assertEqual(loaded.id, volume_target.id)
    self.assertIsNotNone(loaded.node_id)
    with_fields = self.conn.baremetal.get_volume_target(volume_target.id, fields=['uuid', 'extra'])
    self.assertEqual(volume_target.id, with_fields.id)
    self.assertIsNone(with_fields.node_id)
    self.conn.baremetal.delete_volume_target(volume_target, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_volume_target, volume_target.id)