import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_rebuild_with_image_name(self):
    image_name = 'my-custom-image'
    user_image = image_fakes.create_one_image(attrs={'name': image_name})
    self.image_client.find_image.return_value = user_image
    attrs = {'image': {'id': user_image.id}, 'networks': {}, 'adminPass': 'passw0rd'}
    new_server = compute_fakes.create_one_server(attrs=attrs)
    self.server.rebuild.return_value = new_server
    arglist = [self.server.id, '--image', image_name]
    verifylist = [('server', self.server.id), ('image', image_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.servers_mock.get.assert_called_with(self.server.id)
    self.image_client.find_image.assert_called_with(image_name, ignore_missing=False)
    self.image_client.get_image.assert_called_with(user_image.id)
    self.server.rebuild.assert_called_with(user_image, None)