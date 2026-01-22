import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_list_no_options(self):
    arglist = []
    verifylist = [('limit', None), ('sort_key', None), ('sort_dir', None), ('fields', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.cluster_templates_mock.list.assert_called_with(limit=None, sort_dir=None, sort_key=None)
    self.assertEqual(self.columns, columns)
    index = 0
    for d in data:
        self.assertEqual(self.datalist[index], d)
        index += 1