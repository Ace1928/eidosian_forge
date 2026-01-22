from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
def test_volume_set_property(self):
    arglist = ['--property', 'a=b', '--property', 'c=d', self.new_volume.id]
    verifylist = [('property', {'a': 'b', 'c': 'd'}), ('volume', self.new_volume.id), ('bootable', False), ('non_bootable', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.volumes_mock.set_metadata.assert_called_with(self.new_volume.id, parsed_args.property)