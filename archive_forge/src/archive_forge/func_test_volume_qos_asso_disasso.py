import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_qos_asso_disasso(self):
    """Tests associate and disassociate qos with volume type"""
    vol_type1 = uuid.uuid4().hex
    cmd_output = self.openstack('volume type create ' + vol_type1, parse_output=True)
    self.assertEqual(vol_type1, cmd_output['name'])
    self.addCleanup(self.openstack, 'volume type delete ' + vol_type1)
    vol_type2 = uuid.uuid4().hex
    cmd_output = self.openstack('volume type create ' + vol_type2, parse_output=True)
    self.assertEqual(vol_type2, cmd_output['name'])
    self.addCleanup(self.openstack, 'volume type delete ' + vol_type2)
    name = uuid.uuid4().hex
    cmd_output = self.openstack('volume qos create ' + name, parse_output=True)
    self.assertEqual(name, cmd_output['name'])
    self.addCleanup(self.openstack, 'volume qos delete ' + name)
    raw_output = self.openstack('volume qos associate ' + name + ' ' + vol_type1)
    self.assertOutput('', raw_output)
    raw_output = self.openstack('volume qos associate ' + name + ' ' + vol_type2)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume qos show ' + name, parse_output=True)
    types = cmd_output['associations']
    self.assertIn(vol_type1, types)
    self.assertIn(vol_type2, types)
    raw_output = self.openstack('volume qos disassociate ' + '--volume-type ' + vol_type1 + ' ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume qos show ' + name, parse_output=True)
    types = cmd_output['associations']
    self.assertNotIn(vol_type1, types)
    self.assertIn(vol_type2, types)
    raw_output = self.openstack('volume qos associate ' + name + ' ' + vol_type1)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume qos show ' + name, parse_output=True)
    types = cmd_output['associations']
    self.assertIn(vol_type1, types)
    self.assertIn(vol_type2, types)
    raw_output = self.openstack('volume qos disassociate ' + '--all ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume qos show ' + name, parse_output=True)
    self.assertNotIn('associations', cmd_output.keys())