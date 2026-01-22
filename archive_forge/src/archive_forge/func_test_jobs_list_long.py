from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import jobs as api_j
from saharaclient.osc.v2 import jobs as osc_j
from saharaclient.tests.unit.osc.v1 import test_jobs as tj_v1
def test_jobs_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Id', 'Cluster id', 'Job template id', 'Status', 'Start time', 'End time']
    self.assertEqual(expected_columns, columns)
    expected_data = [('j_id', 'cluster_id', 'job_template_id', 'SUCCEEDED', 'start', 'end')]
    self.assertEqual(expected_data, list(data))