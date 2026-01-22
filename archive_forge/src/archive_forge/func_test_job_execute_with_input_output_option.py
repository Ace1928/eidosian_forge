from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import jobs as api_j
from saharaclient.osc.v2 import jobs as osc_j
from saharaclient.tests.unit.osc.v1 import test_jobs as tj_v1
def test_job_execute_with_input_output_option(self):
    arglist = ['--job-template', 'job-template', '--cluster', 'cluster', '--input', 'input', '--output', 'output']
    verifylist = [('job_template', 'job-template'), ('cluster', 'cluster'), ('input', 'input'), ('output', 'output')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.j_mock.create.assert_called_once_with(cluster_id='cluster_id', configs={}, input_id='ds_id', interface=None, is_protected=False, is_public=False, job_template_id='job_template_id', output_id='ds_id')
    arglist = ['--job-template', 'job-template', '--cluster', 'cluster', '--input', 'input']
    verifylist = [('job_template', 'job-template'), ('cluster', 'cluster'), ('input', 'input')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.j_mock.create.assert_called_with(cluster_id='cluster_id', configs={}, input_id='ds_id', interface=None, is_protected=False, is_public=False, job_template_id='job_template_id', output_id=None)
    arglist = ['--job-template', 'job-template', '--cluster', 'cluster']
    verifylist = [('job_template', 'job-template'), ('cluster', 'cluster')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.j_mock.create.assert_called_with(cluster_id='cluster_id', configs={}, input_id=None, interface=None, is_protected=False, is_public=False, job_template_id='job_template_id', output_id=None)