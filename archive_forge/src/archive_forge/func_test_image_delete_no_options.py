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
def test_image_delete_no_options(self):
    images = image_fakes.create_images(count=1)
    arglist = [images[0].id]
    verifylist = [('images', [images[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.image_client.find_image.side_effect = images
    result = self.cmd.take_action(parsed_args)
    self.image_client.delete_image.assert_called_with(images[0].id, store=parsed_args.store, ignore_missing=False)
    self.assertIsNone(result)