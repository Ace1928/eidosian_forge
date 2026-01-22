from heat_integrationtests.functional import functional_base
def test_value_from_nested_stack(self):
    env = {'parameters': {'flavor': self.conf.minimal_instance_type, 'image': self.conf.minimal_image_ref, 'public_net': self.conf.fixed_network_name}}
    self.stack_create(template=template_value_from_nested_stack_main, environment=env, files={'network.yaml': template_value_from_nested_stack_network})