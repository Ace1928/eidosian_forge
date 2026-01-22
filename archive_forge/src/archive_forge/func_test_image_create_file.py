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
def test_image_create_file(self):
    imagefile = tempfile.NamedTemporaryFile(delete=False)
    imagefile.write(b'\x00')
    imagefile.close()
    arglist = ['--file', imagefile.name, '--unprotected' if not self.new_image.is_protected else '--protected', '--public' if self.new_image.visibility == 'public' else '--private', '--property', 'Alpha=1', '--property', 'Beta=2', '--tag', self.new_image.tags[0], '--tag', self.new_image.tags[1], self.new_image.name]
    verifylist = [('filename', imagefile.name), ('is_protected', self.new_image.is_protected), ('visibility', self.new_image.visibility), ('properties', {'Alpha': '1', 'Beta': '2'}), ('tags', self.new_image.tags), ('name', self.new_image.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.create_image.assert_called_with(name=self.new_image.name, allow_duplicates=True, container_format=_image.DEFAULT_CONTAINER_FORMAT, disk_format=_image.DEFAULT_DISK_FORMAT, is_protected=self.new_image.is_protected, visibility=self.new_image.visibility, Alpha='1', Beta='2', tags=self.new_image.tags, filename=imagefile.name)
    self.assertEqual(self.expected_columns, columns)
    self.assertCountEqual(self.expected_data, data)