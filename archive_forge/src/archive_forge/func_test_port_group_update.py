from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_port_group_update(self):
    port_group = self.create_port_group()
    port_group.extra = {'answer': 42}
    port_group = self.conn.baremetal.update_port_group(port_group)
    self.assertEqual({'answer': 42}, port_group.extra)
    port_group = self.conn.baremetal.get_port_group(port_group.id)
    self.assertEqual({'answer': 42}, port_group.extra)