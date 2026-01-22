from io import StringIO
from unittest import mock
from openstackclient.common import project_cleanup
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_project_cleanup_with_skip_resource(self):
    skip_resource = 'block_storage.backup'
    arglist = ['--project', self.project.id, '--skip-resource', skip_resource]
    verifylist = [('skip_resource', [skip_resource])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = None
    with mock.patch('sys.stdin', StringIO('y')):
        result = self.cmd.take_action(parsed_args)
    self.sdk_connect_as_project_mock.assert_called_with(self.project)
    calls = [mock.call(dry_run=True, status_queue=mock.ANY, filters={}, skip_resources=[skip_resource]), mock.call(dry_run=False, status_queue=mock.ANY, filters={})]
    self.project_cleanup_mock.assert_has_calls(calls)
    self.assertIsNone(result)