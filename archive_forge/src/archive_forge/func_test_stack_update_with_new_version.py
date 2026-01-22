import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_with_new_version(self):
    """Update handles new template version in failure.

        If a stack update fails while changing the template version, update is
        able to handle the new version fine.
        """
    stack_identifier = self.stack_create(template=test_template_one_resource)
    template = _change_rsrc_properties(test_template_two_resource, ['test1'], {'fail': True})
    template['heat_template_version'] = '2015-10-15'
    template['resources']['test2']['properties']['value'] = {'list_join': [',', ['a'], ['b']]}
    self.update_stack(stack_identifier, template=template, expected_status='UPDATE_FAILED')
    template = _change_rsrc_properties(template, ['test2'], {'value': 'Test2'})
    template['resources']['test1']['properties']['action_wait_secs'] = {'create': 1}
    self.update_stack(stack_identifier, template=template, expected_status='UPDATE_FAILED')
    self._stack_delete(stack_identifier)