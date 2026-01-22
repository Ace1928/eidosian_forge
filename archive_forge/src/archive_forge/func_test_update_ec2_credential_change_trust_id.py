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
def test_update_ec2_credential_change_trust_id(self):
    """Call ``PATCH /credentials/{credential_id}``."""
    blob, ref = unit.new_ec2_credential(user_id=self.user['id'], project_id=self.project_id)
    blob['trust_id'] = uuid.uuid4().hex
    ref['blob'] = json.dumps(blob)
    r = self.post('/credentials', body={'credential': ref})
    self.assertValidCredentialResponse(r, ref)
    credential_id = r.result.get('credential')['id']
    blob['trust_id'] = uuid.uuid4().hex
    update_ref = {'blob': json.dumps(blob)}
    self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)
    del blob['trust_id']
    update_ref = {'blob': json.dumps(blob)}
    self.patch('/credentials/%(credential_id)s' % {'credential_id': credential_id}, body={'credential': update_ref}, expected_status=http.client.BAD_REQUEST)