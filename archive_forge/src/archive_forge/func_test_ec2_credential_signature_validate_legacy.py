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
def test_ec2_credential_signature_validate_legacy(self):
    """Test signature validation with a legacy v3 ec2 credential."""
    cred_json, _ = self._create_dict_blob_credential()
    cred_blob = json.loads(cred_json)
    self._test_get_token(access=cred_blob['access'], secret=cred_blob['secret'])