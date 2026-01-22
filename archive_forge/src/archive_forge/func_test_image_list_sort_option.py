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
@mock.patch('osc_lib.utils.sort_items')
def test_image_list_sort_option(self, si_mock):
    si_mock.return_value = [copy.deepcopy(self._image)]
    arglist = ['--sort', 'name:asc']
    verifylist = [('sort', 'name:asc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.images.assert_called_with()
    si_mock.assert_called_with([self._image], 'name:asc', str)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, tuple(data))