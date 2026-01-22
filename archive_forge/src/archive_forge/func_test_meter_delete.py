import uuid
from openstackclient.tests.functional.network.v2 import common
def test_meter_delete(self):
    """Test create, delete multiple"""
    name1 = uuid.uuid4().hex
    name2 = uuid.uuid4().hex
    description = 'fakedescription'
    json_output = self.openstack('network meter create ' + ' --description ' + description + ' ' + name1, parse_output=True)
    self.assertEqual(name1, json_output.get('name'))
    self.assertFalse(json_output.get('shared'))
    self.assertEqual('fakedescription', json_output.get('description'))
    json_output_2 = self.openstack('network meter create ' + '--description ' + description + ' ' + name2, parse_output=True)
    self.assertEqual(name2, json_output_2.get('name'))
    self.assertFalse(json_output_2.get('shared'))
    self.assertEqual('fakedescription', json_output_2.get('description'))
    raw_output = self.openstack('network meter delete ' + name1 + ' ' + name2)
    self.assertOutput('', raw_output)