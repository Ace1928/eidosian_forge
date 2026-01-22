from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_volume_connector_list(self):
    node2 = self.create_node(name='test-node')
    self.conn.baremetal.set_node_provision_state(node2, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(node2, 'power off')
    self.conn.baremetal.set_node_provision_state(self.node, 'manage', wait=True)
    self.conn.baremetal.set_node_power_state(self.node, 'power off')
    vc1 = self.create_volume_connector(connector_id='iqn.2018-07.org.openstack:01:d9a514g2c32', node_id=node2.id, type='iqn')
    vc2 = self.create_volume_connector(connector_id='iqn.2017-07.org.openstack:01:d9a51732c4g', node_id=self.node.id, type='iqn')
    vcs = self.conn.baremetal.volume_connectors(node=self.node.id)
    self.assertEqual([v.id for v in vcs], [vc2.id])
    vcs = self.conn.baremetal.volume_connectors(node=node2.id)
    self.assertEqual([v.id for v in vcs], [vc1.id])
    vcs = self.conn.baremetal.volume_connectors(node='test-node')
    self.assertEqual([v.id for v in vcs], [vc1.id])