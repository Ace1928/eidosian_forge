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
def test_volume_create_with_source(self):
    self.volumes_mock.get.return_value = self.new_volume
    arglist = ['--source', self.new_volume.id, self.new_volume.display_name]
    verifylist = [('source', self.new_volume.id), ('name', self.new_volume.display_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.create.assert_called_with(None, None, self.new_volume.id, self.new_volume.display_name, None, None, None, None, None, None, None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)