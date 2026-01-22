import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_subnet_pool_list(self):
    """Test create, list filter"""
    cmd_output = self.openstack('token issue', parse_output=True)
    auth_project_id = cmd_output['project_id']
    cmd_output = self.openstack('project list', parse_output=True)
    admin_project_id = None
    demo_project_id = None
    for p in cmd_output:
        if p['Name'] == 'admin':
            admin_project_id = p['ID']
        if p['Name'] == 'demo':
            demo_project_id = p['ID']
    self.assertIsNotNone(admin_project_id)
    self.assertIsNotNone(demo_project_id)
    self.assertNotEqual(admin_project_id, demo_project_id)
    self.assertEqual(admin_project_id, auth_project_id)
    name1 = uuid.uuid4().hex
    name2 = uuid.uuid4().hex
    cmd_output, pool_prefix = self._subnet_pool_create('--project ' + demo_project_id + ' --no-share ', name1)
    self.addCleanup(self.openstack, 'subnet pool delete ' + name1)
    self.assertEqual(name1, cmd_output['name'])
    self.assertEqual(False, cmd_output['shared'])
    self.assertEqual(demo_project_id, cmd_output['project_id'])
    self.assertEqual([pool_prefix], cmd_output['prefixes'])
    cmd_output, pool_prefix = self._subnet_pool_create(' --share ', name2)
    self.addCleanup(self.openstack, 'subnet pool delete ' + name2)
    self.assertEqual(name2, cmd_output['name'])
    self.assertEqual(True, cmd_output['shared'])
    self.assertEqual(admin_project_id, cmd_output['project_id'])
    self.assertEqual([pool_prefix], cmd_output['prefixes'])
    cmd_output = self.openstack('subnet pool list ' + '--project ' + demo_project_id, parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name1, names)
    self.assertNotIn(name2, names)
    cmd_output = self.openstack('subnet pool list ' + '--share', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertNotIn(name1, names)
    self.assertIn(name2, names)
    cmd_output = self.openstack('subnet pool list ' + '--name ' + name1, parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name1, names)
    self.assertNotIn(name2, names)
    cmd_output = self.openstack('subnet pool list ' + '--long ', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name1, names)
    self.assertIn(name2, names)