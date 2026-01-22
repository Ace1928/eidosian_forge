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
def test_server_list_n_option(self):
    self.data = tuple(((s.id, s.name, s.status, server.AddressesColumn(s.addresses), s.image['id'] if s.image else server.IMAGE_STRING_FOR_BFV, s.flavor['id']) for s in self.servers))
    arglist = ['-n']
    verifylist = [('all_projects', False), ('no_name_lookup', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
    self.image_client.images.assert_not_called()
    self.compute_sdk_client.flavors.assert_not_called()
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, tuple(data))