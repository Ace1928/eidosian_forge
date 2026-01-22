import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_change_in_file_path(self):
    stack_identifier = self.stack_create(template=self.template, files={'nested.yaml': self.nested_templ}, environment=self.env_templ)
    stack = self.client.stacks.get(stack_identifier)
    secret_out1 = self._stack_output(stack, 'secret-out')
    nested_templ_2 = '\nheat_template_version: 2013-05-23\nresources:\n  secret2:\n    type: OS::Heat::RandomString\noutputs:\n  value:\n    value: freddy\n'
    env_templ_2 = '\nresource_registry:\n  "OS::Heat::RandomString": new/nested.yaml\n'
    self.update_stack(stack_identifier, template=self.template, files={'new/nested.yaml': nested_templ_2}, environment=env_templ_2)
    stack = self.client.stacks.get(stack_identifier)
    secret_out2 = self._stack_output(stack, 'secret-out')
    self.assertNotEqual(secret_out1, secret_out2)
    self.assertEqual('freddy', secret_out2)