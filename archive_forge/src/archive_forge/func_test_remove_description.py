import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_remove_description(self):
    server, descr = self._boot_server_with_description()
    self.nova("update %s --description ''" % server.id)
    output = self.nova('show %s' % server.id)
    self.assertEqual('-', self._get_value_from_the_table(output, 'description'))