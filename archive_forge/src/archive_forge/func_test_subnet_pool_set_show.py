import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_subnet_pool_set_show(self):
    """Test create, delete, set, show, unset"""
    name = uuid.uuid4().hex
    new_name = name + '_'
    cmd_output, pool_prefix = self._subnet_pool_create('--default-prefix-length 16 ' + '--min-prefix-length 16 ' + '--max-prefix-length 32 ' + '--description aaaa ' + '--default-quota 10 ', name)
    self.addCleanup(self.openstack, 'subnet pool delete ' + cmd_output['id'])
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual('aaaa', cmd_output['description'])
    self.assertEqual([pool_prefix], cmd_output['prefixes'])
    self.assertEqual(16, cmd_output['default_prefixlen'])
    self.assertEqual(16, cmd_output['min_prefixlen'])
    self.assertEqual(32, cmd_output['max_prefixlen'])
    self.assertEqual(10, cmd_output['default_quota'])
    cmd_output = self.openstack('subnet pool set ' + '--name ' + new_name + ' --description bbbb ' + ' --pool-prefix 10.110.0.0/16 ' + '--default-prefix-length 8 ' + '--min-prefix-length 8 ' + '--max-prefix-length 16 ' + '--default-quota 20 ' + name)
    self.assertOutput('', cmd_output)
    cmd_output = self.openstack('subnet pool show ' + new_name, parse_output=True)
    self.assertEqual(new_name, cmd_output['name'])
    self.assertEqual('bbbb', cmd_output['description'])
    self.assertEqual(sorted(['10.110.0.0/16', pool_prefix]), sorted(cmd_output['prefixes']))
    self.assertEqual(8, cmd_output['default_prefixlen'])
    self.assertEqual(8, cmd_output['min_prefixlen'])
    self.assertEqual(16, cmd_output['max_prefixlen'])
    self.assertEqual(20, cmd_output['default_quota'])