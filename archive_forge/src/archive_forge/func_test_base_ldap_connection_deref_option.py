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
def test_base_ldap_connection_deref_option(self):

    def get_conn(deref_name):
        self.config_fixture.config(group='ldap', alias_dereferencing=deref_name)
        base_ldap = common_ldap.BaseLdap(CONF)
        return base_ldap.get_connection()
    conn = get_conn('default')
    self.assertEqual(ldap.get_option(ldap.OPT_DEREF), conn.get_option(ldap.OPT_DEREF))
    conn = get_conn('always')
    self.assertEqual(ldap.DEREF_ALWAYS, conn.get_option(ldap.OPT_DEREF))
    conn = get_conn('finding')
    self.assertEqual(ldap.DEREF_FINDING, conn.get_option(ldap.OPT_DEREF))
    conn = get_conn('never')
    self.assertEqual(ldap.DEREF_NEVER, conn.get_option(ldap.OPT_DEREF))
    conn = get_conn('searching')
    self.assertEqual(ldap.DEREF_SEARCHING, conn.get_option(ldap.OPT_DEREF))