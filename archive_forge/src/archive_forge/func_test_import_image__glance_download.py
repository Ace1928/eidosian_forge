import copy
import io
import tempfile
from unittest import mock
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.image.v2 import image as _image
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_import_image__glance_download(self):
    arglist = [self.image.name, '--method', 'glance-download', '--remote-region', 'eu/dublin', '--remote-image', 'remote-image-id', '--remote-service-interface', 'private']
    verifylist = [('image', self.image.name), ('import_method', 'glance-download'), ('remote_region', 'eu/dublin'), ('remote_image', 'remote-image-id'), ('remote_service_interface', 'private')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.image_client.import_image.assert_called_once_with(self.image, method='glance-download', uri=None, remote_region='eu/dublin', remote_image_id='remote-image-id', remote_service_interface='private', stores=None, all_stores=None, all_stores_must_succeed=False)