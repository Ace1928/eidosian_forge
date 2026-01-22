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
def test_ec2_delete_credential(self):
    """Test ec2 credential deletion."""
    ec2_cred = self._get_ec2_cred()
    uri = '/'.join([self._get_ec2_cred_uri(), ec2_cred['access']])
    cred_from_credential_api = PROVIDERS.credential_api.list_credentials_for_user(self.user_id, type=CRED_TYPE_EC2)
    self.assertEqual(1, len(cred_from_credential_api))
    self.delete(uri)
    self.assertRaises(exception.CredentialNotFound, PROVIDERS.credential_api.get_credential, cred_from_credential_api[0]['id'])