from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import job_templates as api_j
from saharaclient.osc.v2 import job_templates as osc_j
from saharaclient.tests.unit.osc.v1 import test_job_templates as tjt_v1
def test_job_template_update_all_options(self):
    arglist = ['pig-job', '--name', 'pig-job', '--description', 'descr', '--public', '--protected']
    verifylist = [('job_template', 'pig-job'), ('name', 'pig-job'), ('description', 'descr'), ('is_public', True), ('is_protected', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.job_mock.update.assert_called_once_with('job_id', description='descr', is_protected=True, is_public=True, name='pig-job')
    expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Libs', 'Mains', 'Name', 'Type')
    self.assertEqual(expected_columns, columns)
    expected_data = ('Job for test', 'job_id', False, False, 'lib:lib_id', 'main:main_id', 'pig-job', 'Pig')
    self.assertEqual(expected_data, data)