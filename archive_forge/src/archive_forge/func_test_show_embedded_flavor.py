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
def test_show_embedded_flavor(self):
    arglist = [self.server.name]
    verifylist = [('diagnostics', False), ('topology', False), ('server', self.server.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.server.info['flavor'] = {'ephemeral': 0, 'ram': 512, 'original_name': 'm1.tiny', 'vcpus': 1, 'extra_specs': {}, 'swap': 0, 'disk': 1}
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertIn('original_name', data[2]._value)