from heat.common import identifier
from heat.tests import common
def test_url_path_with_unicode_path(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', u'工')
    self.assertEqual('t/stacks/s/i/%E5%B7%A5', hi.url_path())