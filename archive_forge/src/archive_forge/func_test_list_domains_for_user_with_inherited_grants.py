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
def test_list_domains_for_user_with_inherited_grants(self):
    """Test that inherited roles on the domain are excluded.

        Test Plan:

        - Create two domains, one user, group and role
        - Domain1 is given an inherited user role, Domain2 an inherited
          group role (for a group of which the user is a member)
        - When listing domains for user, neither domain should be returned

        """
    domain1 = unit.new_domain_ref()
    domain1 = PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    domain2 = unit.new_domain_ref()
    domain2 = PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    user = unit.new_user_ref(domain_id=domain1['id'])
    user = PROVIDERS.identity_api.create_user(user)
    group = unit.new_group_ref(domain_id=domain1['id'])
    group = PROVIDERS.identity_api.create_group(group)
    PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    PROVIDERS.assignment_api.create_grant(user_id=user['id'], domain_id=domain1['id'], role_id=role['id'], inherited_to_projects=True)
    PROVIDERS.assignment_api.create_grant(group_id=group['id'], domain_id=domain2['id'], role_id=role['id'], inherited_to_projects=True)
    user_domains = PROVIDERS.assignment_api.list_domains_for_user(user['id'])
    self.assertThat(user_domains, matchers.HasLength(0))