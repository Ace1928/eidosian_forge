import hashlib
import json
import random
from urllib import parse
from swiftclient import utils as swiftclient_utils
import yaml
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_nested_stack_update(self):
    url = self.publish_template(self.nested_template)
    self.template = self.test_template.replace('the.yaml', url)
    stack_identifier = self.stack_create(template=self.template)
    original_nested_id = self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
    stack = self.client.stacks.get(stack_identifier)
    self.assertEqual('bar', self._stack_output(stack, 'output_foo'))
    new_template = yaml.safe_load(self.template)
    props = new_template['Resources']['the_nested']['Properties']
    props['TemplateURL'] = self.publish_template(self.update_template, cleanup=False)
    self.update_stack(stack_identifier, new_template)
    new_nested_id = self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
    self.assertEqual(original_nested_id, new_nested_id)
    updt_stack = self.client.stacks.get(stack_identifier)
    self.assertEqual('foo', self._stack_output(updt_stack, 'output_foo'))