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
def test_server_list_with_changes_before(self):
    self._set_mock_microversion('2.66')
    arglist = ['--changes-before', '2016-03-05T06:27:59Z', '--deleted']
    verifylist = [('changes_before', '2016-03-05T06:27:59Z'), ('deleted', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.kwargs['changes-before'] = '2016-03-05T06:27:59Z'
    self.kwargs['deleted'] = True
    self.compute_sdk_client.servers.assert_called_with(**self.kwargs)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, tuple(data))