import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_list_options(self):
    arglist = ['--limit', '1', '--sort-key', 'key', '--sort-dir', 'asc', '--fields', 'field1,field2']
    verifylist = [('limit', 1), ('sort_key', 'key'), ('sort_dir', 'asc'), ('fields', 'field1,field2')]
    verifycolumns = self.columns + ['field1', 'field2']
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.cluster_templates_mock.list.assert_called_with(limit=1, sort_dir='asc', sort_key='key')
    self.assertEqual(verifycolumns, columns)