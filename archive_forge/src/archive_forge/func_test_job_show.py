from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import jobs as api_j
from saharaclient.osc.v2 import jobs as osc_j
from saharaclient.tests.unit.osc.v1 import test_jobs as tj_v1
def test_job_show(self):
    arglist = ['job_id']
    verifylist = [('job', 'job_id')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.j_mock.get.assert_called_once_with('job_id')
    expected_columns = ('Cluster id', 'End time', 'Engine job id', 'Id', 'Input id', 'Is protected', 'Is public', 'Job template id', 'Output id', 'Start time', 'Status')
    self.assertEqual(expected_columns, columns)
    expected_data = ('cluster_id', 'end', 'engine_job_id', 'j_id', 'input_id', False, False, 'job_template_id', 'output_id', 'start', 'SUCCEEDED')
    self.assertEqual(expected_data, data)