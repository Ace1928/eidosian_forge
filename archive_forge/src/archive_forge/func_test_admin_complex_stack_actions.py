from heat_integrationtests.functional import functional_base
def test_admin_complex_stack_actions(self):
    self.create_stack_setup_admin_client(template=rsg_template)
    updated_template = rsg_template.copy()
    props = updated_template['resources']['random_group']['properties']
    props['count'] = 3
    self.update_stack(self.stack_identifier, template=updated_template)
    self.stack_suspend(self.stack_identifier)
    self.stack_resume(self.stack_identifier)
    resources = {'random_group': 'OS::Heat::ResourceGroup'}
    self.assertEqual(resources, self.list_resources(self.stack_identifier))
    self._stack_delete(self.stack_identifier)