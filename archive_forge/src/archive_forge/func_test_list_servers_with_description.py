import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_list_servers_with_description(self):
    server, descr = self._boot_server_with_description()
    output = self.nova('list --fields description')
    self.assertEqual(server.id, self._get_column_value_from_single_row_table(output, 'ID'))
    self.assertEqual(descr, self._get_column_value_from_single_row_table(output, 'Description'))