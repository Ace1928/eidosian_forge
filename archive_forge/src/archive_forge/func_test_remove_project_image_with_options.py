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
def test_remove_project_image_with_options(self):
    arglist = [self._image.id, self.project.id, '--project-domain', self.domain.id]
    verifylist = [('image', self._image.id), ('project', self.project.id), ('project_domain', self.domain.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.image_client.remove_member.assert_called_with(member=self.project.id, image=self._image.id)
    self.assertIsNone(result)