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
def enable_multi_domain(self):
    default_config = {'ldap': {'url': 'fake://memory', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=example,cn=com'}, 'identity': {'driver': 'ldap'}}
    domain1_config = {'ldap': {'url': 'fake://memory1', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=example,cn=com'}, 'identity': {'driver': 'ldap', 'list_limit': 101}}
    domain2_config = {'ldap': {'url': 'fake://memory', 'user': 'cn=Admin', 'password': 'password', 'suffix': 'cn=myroot,cn=com', 'group_tree_dn': 'ou=UserGroups,dc=myroot,dc=org', 'user_tree_dn': 'ou=Users,dc=myroot,dc=org'}, 'identity': {'driver': 'ldap'}}
    PROVIDERS.domain_config_api.create_config(CONF.identity.default_domain_id, default_config)
    PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], domain1_config)
    PROVIDERS.domain_config_api.create_config(self.domains['domain2']['id'], domain2_config)
    self.config_fixture.config(group='identity', domain_specific_drivers_enabled=True, domain_configurations_from_database=True, list_limit=1000)
    self.config_fixture.config(group='identity_mapping', backward_compatible_ids=False)