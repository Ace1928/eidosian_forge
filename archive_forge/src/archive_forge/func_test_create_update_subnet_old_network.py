from heat_integrationtests.functional import functional_base
def test_create_update_subnet_old_network(self):
    env = {'parameters': {'net_cidr': '11.11.11.0/24'}}
    stack_identifier = self.stack_create(template=template_subnet_old_network, environment=env)
    env = {'parameters': {'net_cidr': '11.11.12.0/24'}}
    self.update_stack(stack_identifier, template=template_subnet_old_network, environment=env)