from os_service_types import service_types
from os_service_types.tests import base
def test_normalize_none(self):
    self.assertIsNone(service_types._normalize_type(None))