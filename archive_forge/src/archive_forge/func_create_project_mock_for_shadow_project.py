import uuid
from unittest import mock
from keystone.assignment.core import Manager as AssignmentApi
from keystone.auth.plugins import mapped
from keystone.exception import ProjectNotFound
from keystone.resource.core import Manager as ResourceApi
from keystone.tests import unit
def create_project_mock_for_shadow_project(self, shadow_project):
    project = shadow_project.copy()
    project['id'] = uuid.uuid4().hex
    return project