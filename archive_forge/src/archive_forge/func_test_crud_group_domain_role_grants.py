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
def test_crud_group_domain_role_grants(self):
    time = datetime.datetime.utcnow()
    with freezegun.freeze_time(time) as frozen_datetime:
        collection_url = '/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': self.domain_id, 'group_id': self.group_id}
        member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=self.role, resource_url=collection_url)
        self.head(collection_url, expected_status=http.client.OK)
        self.delete(member_url)
        frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, expected_length=0, resource_url=collection_url)
        self.head(collection_url, expected_status=http.client.OK)