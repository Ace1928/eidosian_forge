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
@unit.skip_if_cache_disabled('assignment')
def test_delete_grant_from_group_and_project_invalidates_cache(self):
    new_project = unit.new_project_ref(domain_id=self.domain_id)
    PROVIDERS.resource_api.create_project(new_project['id'], new_project)
    collection_url = '/projects/%(project_id)s/groups/%(group_id)s/roles' % {'project_id': new_project['id'], 'group_id': self.group['id']}
    member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
    self.put(member_url)
    self.head(member_url)
    self.get(member_url, expected_status=http.client.NO_CONTENT)
    resp = self.get(collection_url)
    self.assertValidRoleListResponse(resp, ref=self.role, resource_url=collection_url)
    self.delete(member_url)
    resp = self.get(collection_url)
    self.assertListEqual(resp.json_body['roles'], [])