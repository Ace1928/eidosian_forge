from openstack.network.v2 import metering_label_rule
from openstack.tests.unit import base
def test_make_it_source_and_destination(self):
    custom_example = EXAMPLE.copy()
    custom_example['source_ip_prefix'] = '192.168.0.11/32'
    custom_example['destination_ip_prefix'] = '0.0.0.0/0'
    sot = metering_label_rule.MeteringLabelRule(**custom_example)
    self.assertEqual(custom_example['direction'], sot.direction)
    self.assertFalse(sot.is_excluded)
    self.assertEqual(custom_example['id'], sot.id)
    self.assertEqual(custom_example['metering_label_id'], sot.metering_label_id)
    self.assertEqual(custom_example['project_id'], sot.project_id)
    self.assertEqual(custom_example['remote_ip_prefix'], sot.remote_ip_prefix)
    self.assertEqual(custom_example['source_ip_prefix'], sot.source_ip_prefix)
    self.assertEqual(custom_example['destination_ip_prefix'], sot.destination_ip_prefix)