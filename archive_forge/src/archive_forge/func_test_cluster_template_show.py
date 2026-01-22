import copy
from unittest import mock
from unittest.mock import call
from magnumclient.exceptions import InvalidAttribute
from magnumclient.osc.v1 import cluster_templates as osc_ct
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
from osc_lib import exceptions as osc_exceptions
def test_cluster_template_show(self):
    arglist = ['fake-ct-1']
    verifylist = [('cluster-template', 'fake-ct-1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.cluster_templates_mock.get.assert_called_with('fake-ct-1')
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.show_data_list, data)