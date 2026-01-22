import operator
import uuid
from osc_placement.tests.functional import base
def test_fail_if_incorrect_options(self):
    self.assertCommandFailed('Operation or argument is not supported', self.resource_provider_list, aggregate_uuids=['1'])
    self.assertCommandFailed('Operation or argument is not supported', self.resource_provider_list, resources=['1'])