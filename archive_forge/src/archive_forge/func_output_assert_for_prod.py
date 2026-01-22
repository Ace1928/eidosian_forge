from heat_integrationtests.functional import functional_base
def output_assert_for_prod(self, stack_id, bj_prod=True):
    output = self.client.stacks.output_show(stack_id, 'res_value')['output']
    self.assertEqual('prod_res', output['output_value'])
    test_res_value = self.client.stacks.output_show(stack_id, 'test_res_value')['output']
    self.assertEqual('env_is_prod', test_res_value['output_value'])
    prod_resource = self.client.stacks.output_show(stack_id, 'prod_resource')['output']
    self.assertNotEqual('no_prod_res', prod_resource['output_value'])
    test_res_output = self.client.stacks.output_show(stack_id, 'test_res1_value')['output']
    self.assertEqual('no_test_res1', test_res_output['output_value'])
    beijing_prod_res = self.client.stacks.output_show(stack_id, 'beijing_prod_res')['output']
    if bj_prod:
        self.assertNotEqual('no_prod_res', beijing_prod_res['output_value'])
    else:
        self.assertEqual('no_prod_res', beijing_prod_res['output_value'])