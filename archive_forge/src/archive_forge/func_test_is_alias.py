import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_is_alias(self):
    self.assertEqual(self.is_alias, self.service_types.is_alias(self.service_type))