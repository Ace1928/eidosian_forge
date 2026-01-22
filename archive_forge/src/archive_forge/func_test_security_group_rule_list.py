import random
from openstackclient.tests.functional.network.v2 import common
def test_security_group_rule_list(self):
    cmd_output = self.openstack('default security group rule list ', parse_output=True)
    self.assertIn(self.DEFAULT_SG_RULE_ID, [rule['ID'] for rule in cmd_output])