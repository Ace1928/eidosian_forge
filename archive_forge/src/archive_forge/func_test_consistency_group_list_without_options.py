from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_consistency_group_list_without_options(self):
    arglist = []
    verifylist = [('all_projects', False), ('long', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.consistencygroups_mock.list.assert_called_once_with(detailed=True, search_opts={'all_tenants': False})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))