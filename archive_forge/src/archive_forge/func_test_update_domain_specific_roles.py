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
def test_update_domain_specific_roles(self):
    self.domainA_role1['name'] = uuid.uuid4().hex
    self.patch('/roles/%(role_id)s' % {'role_id': self.domainA_role1['id']}, body={'role': self.domainA_role1})
    r = self.get('/roles/%s' % self.domainA_role1['id'])
    self.assertValidRoleResponse(r, self.domainA_role1)