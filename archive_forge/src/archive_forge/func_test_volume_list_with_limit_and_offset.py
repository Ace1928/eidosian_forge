from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.image.v1 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v1 import fakes as volume_fakes
from openstackclient.volume.v1 import volume
def test_volume_list_with_limit_and_offset(self):
    arglist = ['--limit', '2', '--offset', '5']
    verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', None), ('limit', 2), ('offset', 5)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.list.assert_called_once_with(limit=2, search_opts={'offset': 5, 'status': None, 'display_name': None, 'all_tenants': False})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, tuple(data))