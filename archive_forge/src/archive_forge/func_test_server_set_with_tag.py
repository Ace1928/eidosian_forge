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
def test_server_set_with_tag(self):
    self.fake_servers[0].api_version = api_versions.APIVersion('2.26')
    arglist = ['--tag', 'tag1', '--tag', 'tag2', 'foo_vm']
    verifylist = [('tags', ['tag1', 'tag2']), ('server', 'foo_vm')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.fake_servers[0].add_tag.assert_has_calls([mock.call(tag='tag1'), mock.call(tag='tag2')])
    self.assertIsNone(result)