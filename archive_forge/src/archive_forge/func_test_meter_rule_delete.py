import unittest
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_meter_rule_delete(self):
    """test create, delete"""
    json_output = self.openstack('network meter rule create ' + '--remote-ip-prefix 10.0.0.0/8 ' + self.METER_ID, parse_output=True)
    rule_id = json_output.get('id')
    re_ip = json_output.get('remote_ip_prefix')
    self.addCleanup(self.openstack, 'network meter rule delete ' + rule_id)
    self.assertIsNotNone(re_ip)
    self.assertIsNotNone(rule_id)
    self.assertEqual('10.0.0.0/8', re_ip)