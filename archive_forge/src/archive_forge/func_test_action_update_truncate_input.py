import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_update_truncate_input(self):
    input_value = 'very_long_input_parameter_name_that_should_be_truncated'
    act_def = '\n        version: "2.0"\n        action1:\n          input:\n            - {0}\n          base: std.noop\n        '.format(input_value)
    self.create_file('action.yaml', act_def)
    self.action_create('action.yaml')
    updated_act = self.mistral_admin('action-update', params='action.yaml')
    updated_act_info = self.get_item_info(get_from=updated_act, get_by='Name', value='action1')
    self.assertEqual(updated_act_info['Input'][:-3], input_value[:25])