from os_traits.hw.gpu import cuda
from os_traits.tests import base
def test_unknown_sdk_support(self):
    self.assertIsNone(cuda.compute_capabilities_supported('UNKNOWN'))