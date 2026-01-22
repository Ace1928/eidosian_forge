from io import StringIO
from unittest import mock
from openstackclient.common import project_cleanup
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_project_cleanup_with_auth_project(self):
    self.app.client_manager.auth_ref = mock.Mock()
    self.app.client_manager.auth_ref.project_id = self.project.id
    arglist = ['--auth-project']
    verifylist = [('dry_run', False), ('auth_project', True), ('project', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = None
    with mock.patch('sys.stdin', StringIO('y')):
        result = self.cmd.take_action(parsed_args)
    self.sdk_connect_as_project_mock.assert_not_called()
    calls = [mock.call(dry_run=True, status_queue=mock.ANY, filters={}, skip_resources=None), mock.call(dry_run=False, status_queue=mock.ANY, filters={})]
    self.project_cleanup_mock.assert_has_calls(calls)
    self.assertIsNone(result)