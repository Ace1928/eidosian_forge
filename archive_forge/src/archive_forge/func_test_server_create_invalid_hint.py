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
def test_server_create_invalid_hint(self):
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--hint', 'a0cf03a5-d921-4877-bb5c-86d26cf818e1', self.new_server.name]
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])
    arglist = ['--image', 'image1', '--flavor', 'flavor1', '--hint', '=a0cf03a5-d921-4877-bb5c-86d26cf818e1', self.new_server.name]
    self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, [])