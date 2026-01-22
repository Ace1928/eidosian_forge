import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_exceeds_max_length(self):
    template_random_string = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  secret:\n    Type: OS::Heat::RandomString\n    Properties:\n      length: 513\n"
    exc = self.assertRaises(exception.StackValidationFailed, self.create_stack, template_random_string)
    self.assertIn('513 is out of range (min: 1, max: 512)', str(exc))