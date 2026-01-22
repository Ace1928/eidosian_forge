import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
def test_get_project_by_name_returns_not_found(self):
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project_by_name, uuid.uuid4().hex, CONF.identity.default_domain_id)