from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_project_name_from_oslo_config(self):
    self.assertEqual(self.PROJECT_NAME, self.create_simple_middleware()._conf.project)