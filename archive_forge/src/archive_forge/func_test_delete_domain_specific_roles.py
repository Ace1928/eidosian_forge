import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_domain_specific_roles(self):
    self.delete('/roles/%(role_id)s' % {'role_id': self.domainA_role1['id']})
    self.get('/roles/%s' % self.domainA_role1['id'], expected_status=http.client.NOT_FOUND)
    r = self.get('/roles?domain_id=%s' % self.domainA['id'])
    self.assertValidRoleListResponse(r, expected_length=1)
    self.assertRoleInListResponse(r, self.domainA_role2)