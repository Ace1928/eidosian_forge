import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
def test_template_resource_update_template_schema(self):
    stack_identifier = self.stack_create(template=self.main_template, files={'the.yaml': self.initial_tmpl})
    stack = self.client.stacks.get(stack_identifier)
    initial_id = self._stack_output(stack, 'identifier')
    initial_val = self._stack_output(stack, 'value')
    self.update_stack(stack_identifier, self.template, files={'the.yaml': self.provider})
    stack = self.client.stacks.get(stack_identifier)
    self.assertEqual(initial_id, self._stack_output(stack, 'identifier'))
    if self.expect == self.NOCHANGE:
        self.assertEqual(initial_val, self._stack_output(stack, 'value'))
    else:
        self.assertNotEqual(initial_val, self._stack_output(stack, 'value'))