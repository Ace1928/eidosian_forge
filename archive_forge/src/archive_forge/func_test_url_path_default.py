from heat.common import identifier
from heat.tests import common
def test_url_path_default(self):
    hi = identifier.HeatIdentifier('t', 's', 'i')
    self.assertEqual('t/stacks/s/i', hi.url_path())