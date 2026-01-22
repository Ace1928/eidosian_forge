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
def test_volume_unset_image_property_fail(self):
    self.volumes_mock.delete_image_metadata.side_effect = exceptions.CommandError()
    arglist = ['--image-property', 'Alpha', '--property', 'Beta', self.new_volume.id]
    verifylist = [('image_property', ['Alpha']), ('property', ['Beta']), ('volume', self.new_volume.id)]
    parsed_args = self.check_parser(self.cmd_unset, arglist, verifylist)
    try:
        self.cmd_unset.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('One or more of the unset operations failed', str(e))
    self.volumes_mock.delete_image_metadata.assert_called_with(self.new_volume.id, parsed_args.image_property)
    self.volumes_mock.delete_metadata.assert_called_with(self.new_volume.id, parsed_args.property)