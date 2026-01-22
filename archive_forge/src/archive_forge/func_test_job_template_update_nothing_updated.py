from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import job_templates as api_j
from saharaclient.osc.v2 import job_templates as osc_j
from saharaclient.tests.unit.osc.v1 import test_job_templates as tjt_v1
def test_job_template_update_nothing_updated(self):
    arglist = ['pig-job']
    verifylist = [('job_template', 'pig-job')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.job_mock.update.assert_called_once_with('job_id')