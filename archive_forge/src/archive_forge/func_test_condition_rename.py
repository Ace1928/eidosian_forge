from heat_integrationtests.functional import functional_base
def test_condition_rename(self):
    stack_identifier = self.stack_create(template=before_rename_tmpl)
    self.update_stack(stack_identifier, template=after_rename_tmpl)
    self.update_stack(stack_identifier, template=fail_rename_tmpl, expected_status='UPDATE_FAILED')
    self.update_stack(stack_identifier, template=recover_rename_tmpl)