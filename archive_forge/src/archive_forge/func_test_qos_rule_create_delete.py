import uuid
from openstackclient.tests.functional.network.v2 import common
def test_qos_rule_create_delete(self):
    policy_name = uuid.uuid4().hex
    self.openstack('network qos policy create %s' % policy_name)
    self.addCleanup(self.openstack, 'network qos policy delete %s' % policy_name)
    rule = self.openstack('network qos rule create --type bandwidth-limit --max-kbps 10000 --max-burst-kbits 1400 --egress %s' % policy_name, parse_output=True)
    raw_output = self.openstack('network qos rule delete %s %s' % (policy_name, rule['id']))
    self.assertEqual('', raw_output)