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
def test_image_set_options(self):
    arglist = ['--name', 'new-name', '--min-disk', '2', '--min-ram', '4', '--container-format', 'ovf', '--disk-format', 'vmdk', '--project', self.project.name, '--project-domain', self.domain.id, self._image.id]
    verifylist = [('name', 'new-name'), ('min_disk', 2), ('min_ram', 4), ('container_format', 'ovf'), ('disk_format', 'vmdk'), ('project', self.project.name), ('project_domain', self.domain.id), ('image', self._image.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'name': 'new-name', 'owner_id': self.project.id, 'min_disk': 2, 'min_ram': 4, 'container_format': 'ovf', 'disk_format': 'vmdk'}
    self.image_client.update_image.assert_called_with(self._image.id, **kwargs)
    self.assertIsNone(result)