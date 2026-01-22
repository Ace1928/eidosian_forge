import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_rollback(self):
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_update_rollback'})
    stack_identifier = self.stack_create(template=template)
    initial_resources = {'test1': 'OS::Heat::TestResource'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    tmpl_update = _change_rsrc_properties(test_template_two_resource, ['test1', 'test2'], {'value': 'test_update_rollback', 'fail': True})
    self.update_stack(stack_identifier, tmpl_update, expected_status='ROLLBACK_COMPLETE', disable_rollback=False)
    updated_resources = {'test1': 'OS::Heat::TestResource'}
    self.assertEqual(updated_resources, self.list_resources(stack_identifier))