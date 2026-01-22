import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_list_head_users_in_group(self):
    """Call ``GET & HEAD /groups/{group_id}/users``."""
    self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group_id, 'user_id': self.user['id']})
    resource_url = '/groups/%(group_id)s/users' % {'group_id': self.group_id}
    r = self.get(resource_url)
    self.assertValidUserListResponse(r, ref=self.user, resource_url=resource_url)
    self.assertIn('/groups/%(group_id)s/users' % {'group_id': self.group_id}, r.result['links']['self'])
    self.head(resource_url, expected_status=http.client.OK)