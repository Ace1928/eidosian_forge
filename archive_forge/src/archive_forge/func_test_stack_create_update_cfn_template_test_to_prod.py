from heat_integrationtests.functional import functional_base
def test_stack_create_update_cfn_template_test_to_prod(self):
    stack_identifier = self.stack_create(template=cfn_template)
    self.res_assert_for_test(stack_identifier)
    self.output_assert_for_test(stack_identifier)
    parms = {'zone': 'fuzhou'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_test(stack_identifier, fj_zone=True)
    self.output_assert_for_test(stack_identifier)
    parms = {'zone': 'xianyang'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_test(stack_identifier, shannxi_provice=True)
    self.output_assert_for_test(stack_identifier)
    parms = {'env_type': 'prod'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier)
    self.output_assert_for_prod(stack_identifier)
    parms = {'env_type': 'prod', 'zone': 'shanghai'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier, False)
    self.output_assert_for_prod(stack_identifier, False)
    parms = {'env_type': 'prod', 'zone': 'xiamen'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=True)
    self.output_assert_for_prod(stack_identifier, False)
    parms = {'env_type': 'prod', 'zone': 'xianyang'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=False, shannxi_provice=True)
    self.output_assert_for_prod(stack_identifier, False)