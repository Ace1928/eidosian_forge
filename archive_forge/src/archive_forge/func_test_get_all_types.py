import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_get_all_types(self):
    self.assertEqual(self.all_types, self.service_types.get_all_types(self.service_type))