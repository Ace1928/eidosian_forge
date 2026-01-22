import time
import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_type_create_list(self):
    name = uuid.uuid4().hex
    cmd_output = self.openstack('volume type create --private ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'volume type delete ' + name)
    self.assertEqual(name, cmd_output['name'])
    cmd_output = self.openstack('volume type show %s' % name, parse_output=True)
    self.assertEqual(name, cmd_output['name'])
    cmd_output = self.openstack('volume type list', parse_output=True)
    self.assertIn(name, [t['Name'] for t in cmd_output])
    cmd_output = self.openstack('volume type list --default', parse_output=True)
    self.assertEqual(1, len(cmd_output))
    self.assertEqual('lvmdriver-1', cmd_output[0]['Name'])