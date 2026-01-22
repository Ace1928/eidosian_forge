from heat_integrationtests.functional import functional_base
def test_update_resources(self):
    params = {'chain-types': 'OS::Heat::None'}
    stack_id = self.stack_create(template=TEMPLATE_PARAM_DRIVEN, parameters=params)
    nested_id = self.group_nested_identifier(stack_id, 'my-chain')
    expected = {'0': 'OS::Heat::None'}
    found = self.list_resources(nested_id)
    self.assertEqual(expected, found)
    params = {'chain-types': 'OS::Heat::None,OS::Heat::None'}
    self.update_stack(stack_id, template=TEMPLATE_PARAM_DRIVEN, parameters=params)
    expected = {'0': 'OS::Heat::None', '1': 'OS::Heat::None'}
    found = self.list_resources(nested_id)
    self.assertEqual(expected, found)