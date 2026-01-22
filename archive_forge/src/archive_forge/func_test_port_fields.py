from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_port_fields(self):
    self.create_node()
    self.create_port(address='11:22:33:44:55:66')
    result = self.conn.baremetal.ports(fields=['uuid', 'node_id'])
    for item in result:
        self.assertIsNotNone(item.id)
        self.assertIsNone(item.address)