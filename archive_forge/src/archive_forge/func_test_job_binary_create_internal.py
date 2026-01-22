from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import fakes
def test_job_binary_create_internal(self):
    m_open = mock.mock_open()
    with mock.patch('builtins.open', m_open, create=True):
        arglist = ['--name', 'job-binary', '--data', 'filepath']
        verifylist = [('name', 'job-binary'), ('data', 'filepath')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.jb_mock.create.assert_called_once_with(description=None, extra=None, is_protected=False, is_public=False, name='job-binary', url='internal-db://jbi_id')
        self.jbi_mock.create.assert_called_once_with('job-binary', '')