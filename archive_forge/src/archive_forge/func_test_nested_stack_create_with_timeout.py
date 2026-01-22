import hashlib
import json
import random
from urllib import parse
from swiftclient import utils as swiftclient_utils
import yaml
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_nested_stack_create_with_timeout(self):
    url = self.publish_template(self.nested_template)
    self.template = self.test_template.replace('the.yaml', url)
    timeout_template = yaml.safe_load(self.template)
    props = timeout_template['Resources']['the_nested']['Properties']
    props['TimeoutInMinutes'] = '50'
    stack_identifier = self.stack_create(template=timeout_template)
    stack = self.client.stacks.get(stack_identifier)
    self.assert_resource_is_a_stack(stack_identifier, 'the_nested')
    self.assertEqual('bar', self._stack_output(stack, 'output_foo'))