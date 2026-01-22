from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_show_minimal_fields(self):
    rows = ['instance_uuid', 'name', 'uuid']
    node_show = self.openstack('baremetal node show {} --fields {} {}'.format(self.node['uuid'], ' '.join(rows), self.api_version))
    nodes_show_rows = self._get_table_rows(node_show)
    self.assertEqual(set(rows), set(nodes_show_rows))