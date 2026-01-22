from io import StringIO
from unittest import mock
from openstackclient.common import project_cleanup
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_project_cleanup_with_auto_approve(self):
    arglist = ['--project', self.project.id, '--auto-approve']
    verifylist = [('dry_run', False), ('auth_project', False), ('project', self.project.id), ('auto_approve', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = None
    result = self.cmd.take_action(parsed_args)
    self.sdk_connect_as_project_mock.assert_called_with(self.project)
    calls = [mock.call(dry_run=True, status_queue=mock.ANY, filters={}, skip_resources=None), mock.call(dry_run=False, status_queue=mock.ANY, filters={})]
    self.project_cleanup_mock.assert_has_calls(calls)
    self.assertIsNone(result)