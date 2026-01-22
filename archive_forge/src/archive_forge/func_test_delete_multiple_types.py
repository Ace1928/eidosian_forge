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
def test_delete_multiple_types(self):
    arglist = []
    for t in self.volume_types:
        arglist.append(t.id)
    verifylist = [('volume_types', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for t in self.volume_types:
        calls.append(call(t))
    self.volume_types_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)