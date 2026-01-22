import copy
from unittest import mock
from unittest.mock import call
import uuid
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import keypair
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_keypair_list_conflicting_user_options(self):
    arglist = ['--user', identity_fakes.user_name, '--project', identity_fakes.project_name]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, None)