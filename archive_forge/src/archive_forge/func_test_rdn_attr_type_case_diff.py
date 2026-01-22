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
def test_rdn_attr_type_case_diff(self):
    rdn1 = ldap.dn.str2dn('cn=cn1')[0]
    rdn2 = ldap.dn.str2dn('CN=cn1')[0]
    self.assertTrue(common_ldap.is_rdn_equal(rdn1, rdn2))