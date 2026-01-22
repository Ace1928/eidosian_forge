from heat.common import identifier
from heat.tests import common
def test_path_escape(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', ':/')
    self.assertEqual('/:/', hi.path)
    self.assertEqual('t/stacks/s/i/%3A/', hi.url_path())
    self.assertEqual('arn:openstack:heat::t:stacks/s/i/%3A/', hi.arn())