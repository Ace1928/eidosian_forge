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
def test_crud_group_domain_role_grants_no_group(self):
    """Grant role on a domain to a group that doesn't exist.

        When grant a role on a domain to a group that doesn't exist, the server
        returns 404 Not Found for the group.

        """
    group_id = uuid.uuid4().hex
    collection_url = '/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': self.domain_id, 'group_id': group_id}
    member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
    self.put(member_url, expected_status=http.client.NOT_FOUND)
    self.head(member_url, expected_status=http.client.NOT_FOUND)
    self.get(member_url, expected_status=http.client.NOT_FOUND)