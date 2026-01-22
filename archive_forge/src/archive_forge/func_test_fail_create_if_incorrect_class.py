import uuid
from osc_placement.tests.functional import base
def test_fail_create_if_incorrect_class(self):
    self.assertCommandFailed('JSON does not validate', self.resource_class_create, 'fake_class')
    self.assertCommandFailed('JSON does not validate', self.resource_class_create, 'CUSTOM_lower')
    self.assertCommandFailed('JSON does not validate', self.resource_class_create, 'CUSTOM_GPU.INTEL')