import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_add_remove_description_on_existing_server(self):
    server = self._create_server()
    descr = 'Add a description for previously-booted VM.'
    self.nova("update %s --description '%s'" % (server.id, descr))
    output = self.nova('show %s' % server.id)
    self.assertEqual(descr, self._get_value_from_the_table(output, 'description'))
    self.nova("update %s --description ''" % server.id)
    output = self.nova('show %s' % server.id)
    self.assertEqual('-', self._get_value_from_the_table(output, 'description'))