from heat_integrationtests.functional import functional_base
def test_deployments_create_delete_in_progress(self):
    stack_identifier = self.stack_create(template=self.sd_template, enable_cleanup=self.enable_cleanup, expected_status='CREATE_IN_PROGRESS')
    self._wait_for_resource_status(stack_identifier, 'deployment', 'CREATE_IN_PROGRESS')
    nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'deployment')
    group_resources = self.list_group_resources(stack_identifier, 'deployment', minimal=False)
    self.assertEqual(4, len(group_resources))
    self._stack_delete(stack_identifier)
    self._wait_for_stack_status(nested_identifier, 'DELETE_COMPLETE', success_on_not_found=True)