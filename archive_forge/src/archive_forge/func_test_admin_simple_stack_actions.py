from heat_integrationtests.functional import functional_base
def test_admin_simple_stack_actions(self):
    self.create_stack_setup_admin_client()
    updated_template = test_template.copy()
    props = updated_template['resources']['test1']['properties']
    props['value'] = 'new_value'
    self.update_stack(self.stack_identifier, template=updated_template)
    self.stack_suspend(self.stack_identifier)
    self.stack_resume(self.stack_identifier)
    initial_resources = {'test1': 'OS::Heat::TestResource'}
    self.assertEqual(initial_resources, self.list_resources(self.stack_identifier))
    self._stack_delete(self.stack_identifier)