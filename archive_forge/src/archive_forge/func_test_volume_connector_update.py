from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_connector_update(self):
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    volume_connector = self.create_volume_connector(connector_id='iqn.2019-07.org.openstack:03:de45b472c40', node_id=self.node.id, type='iqn')
    volume_connector.extra = {'answer': 42}
    volume_connector = self.conn.baremetal.update_volume_connector(volume_connector)
    self.assertEqual({'answer': 42}, volume_connector.extra)
    volume_connector = self.conn.baremetal.get_volume_connector(volume_connector.id)
    self.assertEqual({'answer': 42}, volume_connector.extra)