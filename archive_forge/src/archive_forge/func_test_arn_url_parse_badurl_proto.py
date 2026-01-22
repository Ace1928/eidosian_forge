from heat.common import identifier
from heat.tests import common
def test_arn_url_parse_badurl_proto(self):
    url = 'htt://1.2.3.4/foo/arn%3Aopenstack%3Aheat%3A%3At%3Asticks/s/i/p'
    self.assertRaises(ValueError, identifier.HeatIdentifier.from_arn_url, url)