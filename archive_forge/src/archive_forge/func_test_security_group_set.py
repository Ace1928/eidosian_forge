import uuid
from openstackclient.tests.functional.network.v2 import common
def test_security_group_set(self):
    other_name = uuid.uuid4().hex
    raw_output = self.openstack('security group set --description NSA --stateless --name ' + other_name + ' ' + self.NAME)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('security group show ' + other_name, parse_output=True)
    self.assertEqual('NSA', cmd_output['description'])
    self.assertFalse(cmd_output['stateful'])