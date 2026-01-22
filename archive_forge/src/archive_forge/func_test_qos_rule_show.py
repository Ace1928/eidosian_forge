import uuid
from openstackclient.tests.functional.network.v2 import common
def test_qos_rule_show(self):
    cmd_output = self.openstack('network qos rule show %s %s' % (self.QOS_POLICY_NAME, self.RULE_ID), parse_output=True)
    self.assertEqual(self.RULE_ID, cmd_output['id'])