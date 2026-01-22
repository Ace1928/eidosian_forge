from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_connector_patch(self):
    vol_conn_id = 'iqn.2020-07.org.openstack:04:de45b472c40'
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    volume_connector = self.create_volume_connector(connector_id=vol_conn_id, node_id=self.node.id, type='iqn')
    volume_connector = self.conn.baremetal.patch_volume_connector(volume_connector, dict(path='/extra/answer', op='add', value=42))
    self.assertEqual({'answer': 42}, volume_connector.extra)
    self.assertEqual(vol_conn_id, volume_connector.connector_id)
    volume_connector = self.conn.baremetal.get_volume_connector(volume_connector.id)
    self.assertEqual({'answer': 42}, volume_connector.extra)