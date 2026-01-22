import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
def test_storing_null_domain_id_in_project_ref(self):
    """Test the special storage of domain_id=None in sql resource driver.

        The resource driver uses a special value in place of None for domain_id
        in the project record. This shouldn't escape the driver. Hence we test
        the interface to ensure that you can store a domain_id of None, and
        that any special value used inside the driver does not escape through
        the interface.

        """
    spoiler_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.resource_api.create_project(spoiler_project['id'], spoiler_project)
    project = unit.new_project_ref(domain_id=None, is_domain=True)
    project = PROVIDERS.resource_api.create_project(project['id'], project)
    ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertDictEqual(project, ref)
    ref = PROVIDERS.resource_api.get_project_by_name(project['name'], None)
    self.assertDictEqual(project, ref)
    project2 = unit.new_project_ref(domain_id=None, is_domain=True)
    project2 = PROVIDERS.resource_api.create_project(project2['id'], project2)
    hints = driver_hints.Hints()
    hints.add_filter('domain_id', None)
    refs = PROVIDERS.resource_api.list_projects(hints)
    self.assertThat(refs, matchers.HasLength(2 + self.domain_count))
    self.assertIn(project, refs)
    self.assertIn(project2, refs)
    project['name'] = uuid.uuid4().hex
    PROVIDERS.resource_api.update_project(project['id'], project)
    ref = PROVIDERS.resource_api.get_project(project['id'])
    self.assertDictEqual(project, ref)
    project['enabled'] = False
    PROVIDERS.resource_api.update_project(project['id'], project)
    PROVIDERS.resource_api.delete_project(project['id'])
    self.assertRaises(exception.ProjectNotFound, PROVIDERS.resource_api.get_project, project['id'])