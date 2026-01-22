import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
@test.requires_convergence
def test_stack_update_replace_manual_rollback(self):
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'update_replace_value': '1'})
    stack_identifier = self.stack_create(template=template)
    original_resource_id = self.get_physical_resource_id(stack_identifier, 'test1')
    tmpl_update = _change_rsrc_properties(test_template_one_resource, ['test1'], {'update_replace_value': '2', 'fail': True})
    self.update_stack(stack_identifier, tmpl_update, expected_status='UPDATE_FAILED', disable_rollback=True)
    self.update_stack(stack_identifier, template)
    final_resource_id = self.get_physical_resource_id(stack_identifier, 'test1')
    self.assertEqual(original_resource_id, final_resource_id)