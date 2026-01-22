import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_vif_negative(self):
    uuid = '5c9dcd04-2073-49bc-9618-99ae634d8971'
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.attach_vif_to_node, uuid, self.vif_id)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.list_node_vifs, uuid)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.detach_vif_from_node, uuid, self.vif_id, ignore_missing=False)