import hashlib
import json
from unittest import mock
import uuid
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_db import exception as oslo_db_exception
from testtools import matchers
import urllib
from keystone.api import ec2tokens
from keystone.common import provider_api
from keystone.common import utils
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone import oauth1
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_update_credential_non_owner(self):
    """Call ``PATCH /credentials/{credential_id}``."""
    alt_user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
    alt_user_id = alt_user['id']
    alt_project = unit.new_project_ref(domain_id=self.domain_id)
    alt_project_id = alt_project['id']
    PROVIDERS.resource_api.create_project(alt_project['id'], alt_project)
    alt_role = unit.new_role_ref(name='reader')
    alt_role_id = alt_role['id']
    PROVIDERS.role_api.create_role(alt_role_id, alt_role)
    PROVIDERS.assignment_api.add_role_to_user_and_project(alt_user_id, alt_project_id, alt_role_id)
    auth = self.build_authentication_request(user_id=alt_user_id, password=alt_user['password'], project_id=alt_project_id)
    ref = unit.new_credential_ref(user_id=alt_user_id, project_id=alt_project_id)
    r = self.post('/credentials', auth=auth, body={'credential': ref})
    self.assertValidCredentialResponse(r, ref)
    credential_id = r.result.get('credential')['id']
    update_ref = {'user_id': self.user_id, 'project_id': self.project_id}
    self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, expected_status=403, auth=auth, body={'credential': update_ref})