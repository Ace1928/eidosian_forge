import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_is_known(self):
    self.assertEqual(self.is_known, self.service_types.is_known(self.service_type))