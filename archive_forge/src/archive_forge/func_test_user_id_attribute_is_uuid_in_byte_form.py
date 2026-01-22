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
def test_user_id_attribute_is_uuid_in_byte_form(self):
    results = [('cn=alice,dc=example,dc=com', {'cn': [b'cn=alice'], 'objectGUID': [b'\xdd\xd8Rt\xee]bA\x8e(\xe39\x0b\xe1\xf8\xe8'], 'email': [uuid.uuid4().hex], 'sn': [uuid.uuid4().hex]})]
    py_result = common_ldap.convert_ldap_result(results)
    exp_object_guid = '7452d8dd-5dee-4162-8e28-e3390be1f8e8'
    self.assertEqual(exp_object_guid, py_result[0][1]['objectGUID'][0])