from heat.common import identifier
from heat.tests import common
def test_id_escape(self):
    hi = identifier.HeatIdentifier('t', 's', ':/')
    self.assertEqual(':/', hi.stack_id)
    self.assertEqual('t/stacks/s/%3A%2F', hi.url_path())
    self.assertEqual('arn:openstack:heat::t:stacks/s/%3A%2F', hi.arn())