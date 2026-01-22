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
def test_escape_member_dn(self):
    object_id = uuid.uuid4().hex
    driver = PROVIDERS.identity_api._select_identity_driver(CONF.identity.default_domain_id)
    mixin_impl = driver.user
    sample_dn = 'cn=foo)bar'
    sample_dn_filter_esc = 'cn=foo\\29bar'
    mixin_impl.tree_dn = sample_dn
    exp_filter = '(%s=%s=%s,%s)' % (mixin_impl.member_attribute, mixin_impl.id_attr, object_id, sample_dn_filter_esc)
    with mixin_impl.get_connection() as conn:
        m = self.useFixture(fixtures.MockPatchObject(conn, 'search_s')).mock
        mixin_impl._is_id_enabled(object_id, conn)
        self.assertEqual(exp_filter, m.call_args[0][2])