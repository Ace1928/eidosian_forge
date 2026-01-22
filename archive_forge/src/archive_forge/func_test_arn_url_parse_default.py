from heat.common import identifier
from heat.tests import common
def test_arn_url_parse_default(self):
    url = self.url_prefix + 'arn%3Aopenstack%3Aheat%3A%3At%3Astacks/s/i'
    hi = identifier.HeatIdentifier.from_arn_url(url)
    self.assertEqual('t', hi.tenant)
    self.assertEqual('s', hi.stack_name)
    self.assertEqual('i', hi.stack_id)
    self.assertEqual('', hi.path)