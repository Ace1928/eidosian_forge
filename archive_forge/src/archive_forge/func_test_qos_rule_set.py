import uuid
from openstackclient.tests.functional.network.v2 import common
def test_qos_rule_set(self):
    self.openstack('network qos rule set --max-kbps 15000 --max-burst-kbits 1800 --ingress %s %s' % (self.QOS_POLICY_NAME, self.RULE_ID))
    cmd_output = self.openstack('network qos rule show %s %s' % (self.QOS_POLICY_NAME, self.RULE_ID), parse_output=True)
    self.assertEqual(15000, cmd_output['max_kbps'])
    self.assertEqual(1800, cmd_output['max_burst_kbps'])
    self.assertEqual('ingress', cmd_output['direction'])