from heat_integrationtests.functional import functional_base
def test_stack_create_update_cfn_template_prod_to_test(self):
    parms = {'env_type': 'prod'}
    stack_identifier = self.stack_create(template=cfn_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier)
    self.output_assert_for_prod(stack_identifier)
    parms = {'zone': 'xiamen', 'env_type': 'prod'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=True)
    self.output_assert_for_prod(stack_identifier, bj_prod=False)
    parms = {'zone': 'xianyang', 'env_type': 'prod'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=False, shannxi_provice=True)
    self.output_assert_for_prod(stack_identifier, bj_prod=False)
    parms = {'zone': 'shanghai', 'env_type': 'prod'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=False, shannxi_provice=False)
    self.output_assert_for_prod(stack_identifier, bj_prod=False)
    parms = {'env_type': 'test'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_test(stack_identifier)
    self.output_assert_for_test(stack_identifier)
    parms = {'env_type': 'test', 'zone': 'fuzhou'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_test(stack_identifier, fj_zone=True)
    self.output_assert_for_test(stack_identifier)
    parms = {'env_type': 'test', 'zone': 'xianyang'}
    self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
    self.res_assert_for_test(stack_identifier, fj_zone=False, shannxi_provice=True)
    self.output_assert_for_test(stack_identifier)