from heat.common import identifier
from heat.tests import common
def test_arn_url(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', 'p')
    self.assertEqual('/arn%3Aopenstack%3Aheat%3A%3At%3Astacks/s/i/p', hi.arn_url_path())