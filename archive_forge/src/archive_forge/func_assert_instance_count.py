import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
def assert_instance_count(self, stack, expected_count):
    inst_list = self._stack_output(stack, 'InstanceList')
    self.assertEqual(expected_count, len(inst_list.split(',')))