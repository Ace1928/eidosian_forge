from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_port_list(self):
    node2 = self.create_node(name='test-node')
    port1 = self.create_port(address='11:22:33:44:55:66', node_id=node2.id)
    port2 = self.create_port(address='11:22:33:44:55:77', node_id=self.node.id)
    ports = self.conn.baremetal.ports(address='11:22:33:44:55:77')
    self.assertEqual([p.id for p in ports], [port2.id])
    ports = self.conn.baremetal.ports(node=node2.id)
    self.assertEqual([p.id for p in ports], [port1.id])
    ports = self.conn.baremetal.ports(node='test-node')
    self.assertEqual([p.id for p in ports], [port1.id])