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
@mock.patch('openstackclient.compute.v2.server.io.open')
def test_server_create_userdata(self, mock_open):
    mock_file = mock.Mock(name='File')
    mock_open.return_value = mock_file
    mock_open.read.return_value = '#!/bin/sh'
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--user-data', 'userdata.sh', self.new_server.name]
    verifylist = [('image', 'image1'), ('flavor', 'flavor1'), ('user_data', 'userdata.sh'), ('config_drive', False), ('server_name', self.new_server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    mock_open.assert_called_with('userdata.sh')
    mock_file.close.assert_called_with()
    kwargs = dict(meta=None, files={}, reservation_id=None, min_count=1, max_count=1, security_groups=[], userdata=mock_file, key_name=None, availability_zone=None, admin_pass=None, block_device_mapping_v2=[], nics=[], scheduler_hints={}, config_drive=None)
    self.servers_mock.create.assert_called_with(self.new_server.name, self.image, self.flavor, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist(), data)