import os
import tempfile
from unittest import mock
import uuid
import fixtures
import ldap.dn
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception as ks_exception
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import fakeldap
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
def test_certdir_trust_tls(self):
    certdir = self.useFixture(fixtures.TempDir()).path
    self.config_fixture.config(group='ldap', url='ldap://localhost', use_tls=True, tls_cacertdir=certdir)
    self._init_ldap_connection(CONF)
    self.assertEqual(certdir, ldap.get_option(ldap.OPT_X_TLS_CACERTDIR))