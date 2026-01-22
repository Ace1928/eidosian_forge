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
def test_user_description_attribute_mapping(self):
    self.config_fixture.config(group='ldap', user_description_attribute='displayName')
    self.load_backends()
    user = self.new_user_ref(domain_id=CONF.identity.default_domain_id, displayName=uuid.uuid4().hex)
    description = user['displayName']
    user = PROVIDERS.identity_api.create_user(user)
    res = PROVIDERS.identity_api.driver.user.get_all()
    new_user = [u for u in res if u['id'] == user['id']][0]
    self.assertThat(new_user['description'], matchers.Equals(description))