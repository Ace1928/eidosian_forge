import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workflow_update_truncate_input(self):
    input_value = 'very_long_input_parameter_name_that_should_be_truncated'
    wf_def = '\n        version: "2.0"\n        workflow1:\n          input:\n            - {0}\n          tasks:\n            task1:\n              action: std.noop\n        '.format(input_value)
    self.create_file('wf.yaml', wf_def)
    self.workflow_create('wf.yaml')
    updated_wf = self.mistral_admin('workflow-update', params='wf.yaml')
    updated_wf_info = self.get_item_info(get_from=updated_wf, get_by='Name', value='workflow1')
    self.assertEqual(updated_wf_info['Input'][:-3], input_value[:25])