import copy
import datetime
import os
import tempfile
from oslotest import base
import os_service_types.service_types
def test_empty_project_error(self):
    if not self.project:
        self.assertRaises(ValueError, self.service_types.get_service_data_for_project, self.project)