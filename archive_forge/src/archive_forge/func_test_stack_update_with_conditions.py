import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_with_conditions(self):
    test3 = {'type': 'OS::Heat::TestResource', 'condition': 'cond1', 'properties': {'value': 'foo'}}
    self._test_conditional(test3)