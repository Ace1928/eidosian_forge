import uuid
from openstackclient.tests.functional.object.v1 import common
def test_container_list(self):
    opts = self.get_opts(['Name'])
    raw_output = self.openstack('container list' + opts)
    self.assertIn(self.NAME, raw_output)