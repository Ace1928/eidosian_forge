import uuid
from openstackclient.tests.functional.network.v2 import common
def test_meter_show(self):
    """Test create, show, delete"""
    name1 = uuid.uuid4().hex
    description = 'fakedescription'
    json_output = self.openstack('network meter create ' + ' --description ' + description + ' ' + name1, parse_output=True)
    meter_id = json_output.get('id')
    self.addCleanup(self.openstack, 'network meter delete ' + name1)
    json_output = self.openstack('network meter show ' + meter_id, parse_output=True)
    self.assertFalse(json_output.get('shared'))
    self.assertEqual('fakedescription', json_output.get('description'))
    self.assertEqual(name1, json_output.get('name'))
    json_output = self.openstack('network meter show ' + name1, parse_output=True)
    self.assertEqual(meter_id, json_output.get('id'))
    self.assertFalse(json_output.get('shared'))
    self.assertEqual('fakedescription', json_output.get('description'))