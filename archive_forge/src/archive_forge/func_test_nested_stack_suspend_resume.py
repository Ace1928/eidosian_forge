import hashlib
import json
import random
from urllib import parse
from swiftclient import utils as swiftclient_utils
import yaml
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_nested_stack_suspend_resume(self):
    url = self.publish_template(self.nested_template)
    self.template = self.test_template.replace('the.yaml', url)
    stack_identifier = self.stack_create(template=self.template)
    self.stack_suspend(stack_identifier)
    self.stack_resume(stack_identifier)