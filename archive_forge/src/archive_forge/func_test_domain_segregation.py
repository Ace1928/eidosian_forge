import copy
from unittest import mock
import uuid
import fixtures
import http.client
import ldap
from oslo_log import versionutils
import pkg_resources
from testtools import matchers
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.identity.backends import ldap as ldap_identity
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends import sql as sql_identity
from keystone.identity.mapping_backends import mapping as map
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.resource import test_backends as resource_tests
def test_domain_segregation(self):
    """Test that separate configs have segregated the domain.

        Test Plan:

        - Users were created in each domain as part of setup, now make sure
          you can only find a given user in its relevant domain/backend
        - Make sure that for a backend that supports multiple domains
          you can get the users via any of its domains

        """
    users = self.create_users_across_domains()
    self.check_user(users['user0'], self.domain_default['id'], http.client.OK)
    self.check_user(users['user0'], self.domains['domain1']['id'], exception.UserNotFound)
    self.check_user(users['user1'], self.domains['domain1']['id'], http.client.OK)
    self.check_user(users['user1'], self.domain_default['id'], exception.UserNotFound)
    self.assertThat(PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain1']['id']), matchers.HasLength(1))