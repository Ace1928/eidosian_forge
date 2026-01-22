import uuid
from openstackclient.tests.functional import base
def test_flavor_properties(self):
    """Test create defaults, list filters, delete"""
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('flavor create ' + '--id qaz ' + '--ram 123 ' + '--disk 20 ' + '--private ' + '--property a=first ' + '--property b=second ' + name1, parse_output=True)
    self.addCleanup(self.openstack, 'flavor delete ' + name1)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual('qaz', cmd_output['id'])
    self.assertEqual(name1, cmd_output['name'])
    self.assertEqual(123, cmd_output['ram'])
    self.assertEqual(20, cmd_output['disk'])
    self.assertFalse(cmd_output['os-flavor-access:is_public'])
    self.assertDictEqual({'a': 'first', 'b': 'second'}, cmd_output['properties'])
    raw_output = self.openstack('flavor set ' + "--property a='third and 10' " + '--property g=fourth ' + name1)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('flavor show ' + name1, parse_output=True)
    self.assertEqual('qaz', cmd_output['id'])
    self.assertEqual('third and 10', cmd_output['properties']['a'])
    self.assertEqual('second', cmd_output['properties']['b'])
    self.assertEqual('fourth', cmd_output['properties']['g'])
    raw_output = self.openstack('flavor unset ' + '--property b ' + name1)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('flavor show ' + name1, parse_output=True)
    self.assertNotIn('b', cmd_output['properties'])