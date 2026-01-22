from heat.common import identifier
from heat.tests import common
def test_arn_url_parse_missing_field(self):
    url = self.url_prefix + 'arn%3Aopenstack%3Aheat%3A%3At%3Asticks/s/'
    self.assertRaises(ValueError, identifier.HeatIdentifier.from_arn_url, url)