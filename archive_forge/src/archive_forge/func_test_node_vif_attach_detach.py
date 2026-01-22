import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_vif_attach_detach(self):
    self.conn.baremetal.attach_vif_to_node(self.node, self.vif_id)
    self.conn.baremetal.list_node_vifs(self.node)
    res = self.conn.baremetal.detach_vif_from_node(self.node, self.vif_id, ignore_missing=False)
    self.assertTrue(res)