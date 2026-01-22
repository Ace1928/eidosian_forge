from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_consistency_group_set_name(self):
    new_name = 'new_name'
    arglist = ['--name', new_name, self.consistency_group.id]
    verifylist = [('name', new_name), ('description', None), ('consistency_group', self.consistency_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'name': new_name}
    self.consistencygroups_mock.update.assert_called_once_with(self.consistency_group.id, **kwargs)
    self.assertIsNone(result)