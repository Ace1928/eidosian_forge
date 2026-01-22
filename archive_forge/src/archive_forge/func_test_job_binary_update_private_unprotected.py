from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
def test_job_binary_update_private_unprotected(self):
    arglist = ['job-binary', '--private', '--unprotected']
    verifylist = [('job_binary', 'job-binary'), ('is_public', False), ('is_protected', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.jb_mock.update.assert_called_once_with('jb_id', {'is_public': False, 'is_protected': False})