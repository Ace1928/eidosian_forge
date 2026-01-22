from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_project_name_from_local_config(self):
    project_name = uuid.uuid4().hex
    middleware = self.create_simple_middleware(project=project_name)
    self.assertEqual(project_name, middleware._conf.project)