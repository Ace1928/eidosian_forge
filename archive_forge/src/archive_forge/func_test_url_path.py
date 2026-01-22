from heat.common import identifier
from heat.tests import common
def test_url_path(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', 'p')
    self.assertEqual('t/stacks/s/i/p', hi.url_path())