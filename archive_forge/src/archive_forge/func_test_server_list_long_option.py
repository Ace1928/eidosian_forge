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
def test_server_list_long_option(self):
    self.data = tuple(((s.id, s.name, s.status, getattr(s, 'task_state'), server.PowerStateColumn(getattr(s, 'power_state')), server.AddressesColumn(s.addresses), self.image.name if s.image else server.IMAGE_STRING_FOR_BFV, s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, self.flavor.name, s.flavor['id'], getattr(s, 'availability_zone'), server.HostColumn(getattr(s, 'hypervisor_hostname')), format_columns.DictColumn(s.metadata)) for s in self.servers))
    arglist = ['--long']
    verifylist = [('all_projects', False), ('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
    image_ids = {s.image['id'] for s in self.servers if s.image}
    self.image_client.images.assert_called_once_with(id=f'in:{','.join(image_ids)}')
    self.compute_sdk_client.flavors.assert_called_once_with(is_public=None)
    self.assertEqual(self.columns_long, columns)
    self.assertEqual(self.data, tuple(data))