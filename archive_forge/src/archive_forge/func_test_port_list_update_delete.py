from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_port_list_update_delete(self):
    self.create_port(address='11:22:33:44:55:66', node_id=self.node.id, extra={'foo': 'bar'})
    port = next(self.conn.baremetal.ports(details=True, address='11:22:33:44:55:66'))
    self.assertEqual(port.extra, {'foo': 'bar'})
    self.conn.baremetal.update_port(port, extra={'foo': 42})
    self.conn.baremetal.delete_port(port, ignore_missing=False)