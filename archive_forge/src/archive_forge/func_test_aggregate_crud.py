import uuid
from openstackclient.tests.functional import base
def test_aggregate_crud(self):
    """Test create, delete multiple"""
    name1 = uuid.uuid4().hex
    self.addCleanup(self.openstack, 'aggregate delete ' + name1, fail_ok=True)
    cmd_output = self.openstack('aggregate create ' + '--zone nova ' + '--property a=b ' + name1, parse_output=True)
    self.assertEqual(name1, cmd_output['name'])
    self.assertEqual('nova', cmd_output['availability_zone'])
    self.assertIn('a', cmd_output['properties'])
    cmd_output = self.openstack('aggregate show ' + name1, parse_output=True)
    self.assertEqual(name1, cmd_output['name'])
    name2 = uuid.uuid4().hex
    self.addCleanup(self.openstack, 'aggregate delete ' + name2, fail_ok=True)
    cmd_output = self.openstack('aggregate create ' + '--zone external ' + name2, parse_output=True)
    self.assertEqual(name2, cmd_output['name'])
    self.assertEqual('external', cmd_output['availability_zone'])
    cmd_output = self.openstack('aggregate show ' + name2, parse_output=True)
    self.assertEqual(name2, cmd_output['name'])
    name3 = uuid.uuid4().hex
    self.addCleanup(self.openstack, 'aggregate delete ' + name3, fail_ok=True)
    raw_output = self.openstack('aggregate set ' + '--name ' + name3 + ' ' + '--zone internal ' + '--no-property ' + '--property c=d ' + name1)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('aggregate show ' + name3, parse_output=True)
    self.assertEqual(name3, cmd_output['name'])
    self.assertEqual('internal', cmd_output['availability_zone'])
    self.assertIn('c', cmd_output['properties'])
    self.assertNotIn('a', cmd_output['properties'])
    cmd_output = self.openstack('aggregate list', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name3, names)
    self.assertIn(name2, names)
    zones = [x['Availability Zone'] for x in cmd_output]
    self.assertIn('external', zones)
    self.assertIn('internal', zones)
    cmd_output = self.openstack('aggregate list --long', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name3, names)
    self.assertIn(name2, names)
    zones = [x['Availability Zone'] for x in cmd_output]
    self.assertIn('external', zones)
    self.assertIn('internal', zones)
    properties = [x['Properties'] for x in cmd_output]
    self.assertNotIn({'a': 'b'}, properties)
    self.assertIn({'c': 'd'}, properties)
    raw_output = self.openstack('aggregate unset ' + '--property c ' + name3)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('aggregate show ' + name3, parse_output=True)
    self.assertNotIn("c='d'", cmd_output['properties'])
    del_output = self.openstack('aggregate delete ' + name3 + ' ' + name2)
    self.assertOutput('', del_output)