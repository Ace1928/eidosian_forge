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
def test_ec2_cannot_get_non_ec2_credential(self):
    access_key = uuid.uuid4().hex
    cred_id = utils.hash_access_key(access_key)
    non_ec2_cred = unit.new_credential_ref(user_id=self.user_id, project_id=self.project_id)
    non_ec2_cred['id'] = cred_id
    PROVIDERS.credential_api.create_credential(cred_id, non_ec2_cred)
    uri = '/'.join([self._get_ec2_cred_uri(), access_key])
    self.get(uri, expected_status=http.client.UNAUTHORIZED)