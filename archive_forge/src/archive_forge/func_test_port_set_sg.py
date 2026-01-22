import uuid
from openstackclient.tests.functional.network.v2 import common
def test_port_set_sg(self):
    """Test create, set, show, delete"""
    sg_name1 = uuid.uuid4().hex
    json_output = self.openstack('security group create %s' % sg_name1, parse_output=True)
    sg_id1 = json_output.get('id')
    self.addCleanup(self.openstack, 'security group delete %s' % sg_id1)
    sg_name2 = uuid.uuid4().hex
    json_output = self.openstack('security group create %s' % sg_name2, parse_output=True)
    sg_id2 = json_output.get('id')
    self.addCleanup(self.openstack, 'security group delete %s' % sg_id2)
    name = uuid.uuid4().hex
    json_output = self.openstack('port create --network %s --security-group %s %s' % (self.NETWORK_NAME, sg_name1, name), parse_output=True)
    id1 = json_output.get('id')
    self.addCleanup(self.openstack, 'port delete %s' % id1)
    self.assertEqual(name, json_output.get('name'))
    self.assertEqual([sg_id1], json_output.get('security_group_ids'))
    raw_output = self.openstack('port set --security-group %s %s' % (sg_name2, name))
    self.assertOutput('', raw_output)
    json_output = self.openstack('port show %s' % name, parse_output=True)
    self.assertEqual(name, json_output.get('name'))
    self.assertIsInstance(json_output.get('security_group_ids'), list)
    self.assertEqual(sorted([sg_id1, sg_id2]), sorted(json_output.get('security_group_ids')))
    raw_output = self.openstack('port unset --security-group %s %s' % (sg_id1, id1))
    self.assertOutput('', raw_output)
    json_output = self.openstack('port show %s' % name, parse_output=True)
    self.assertEqual([sg_id2], json_output.get('security_group_ids'))