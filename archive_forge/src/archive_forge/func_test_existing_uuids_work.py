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
def test_existing_uuids_work(self):
    """Test that 'uni-domain' created IDs still work.

        Throwing the switch to domain-specific backends should not cause
        existing identities to be inaccessible via ID.

        """
    userA = unit.create_user(PROVIDERS.identity_api, self.domain_default['id'])
    userB = unit.create_user(PROVIDERS.identity_api, self.domains['domain1']['id'])
    userC = unit.create_user(PROVIDERS.identity_api, self.domains['domain3']['id'])
    PROVIDERS.identity_api.get_user(userA['id'])
    PROVIDERS.identity_api.get_user(userB['id'])
    PROVIDERS.identity_api.get_user(userC['id'])