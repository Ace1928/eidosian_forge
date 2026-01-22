import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_get_service_type_permissive(self):
    self.assertEqual(self.official or self.service_type, self.service_types.get_service_type(self.service_type, permissive=True))