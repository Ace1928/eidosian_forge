from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_type
def test_type_unset_failed_with_missing_volume_type_argument(self):
    arglist = ['--project', 'identity_fakes.project_id']
    verifylist = [('project', 'identity_fakes.project_id')]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)