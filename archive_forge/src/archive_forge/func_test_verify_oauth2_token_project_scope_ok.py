import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_verify_oauth2_token_project_scope_ok(self):
    cache_on_issue = CONF.token.cache_on_issue
    caching = CONF.token.caching
    self._create_mapping()
    user, user_domain, _ = self._create_project_user()
    *_, client_cert, _ = self._create_certificates(root_dn=unit.create_dn(common_name='root'), client_dn=unit.create_dn(common_name=user['name'], user_id=user['id'], email_address=user['email'], organization_name=user_domain['name'], domain_component=user_domain['id']))
    cert_content = self._get_cert_content(client_cert)
    CONF.token.cache_on_issue = False
    CONF.token.caching = False
    resp = self._get_oauth2_access_token(user['id'], cert_content)
    json_resp = json.loads(resp.body)
    self.assertIn('access_token', json_resp)
    self.assertEqual('Bearer', json_resp['token_type'])
    self.assertEqual(3600, json_resp['expires_in'])
    verify_resp = self.get('/auth/tokens', headers={'X-Subject-Token': json_resp['access_token'], 'X-Auth-Token': json_resp['access_token']}, expected_status=http.client.OK)
    self.assertIn('token', verify_resp.result)
    self.assertIn('oauth2_credential', verify_resp.result['token'])
    self.assertIn('roles', verify_resp.result['token'])
    self.assertIn('project', verify_resp.result['token'])
    self.assertIn('catalog', verify_resp.result['token'])
    check_oauth2 = verify_resp.result['token']['oauth2_credential']
    self.assertEqual(utils.get_certificate_thumbprint(cert_content), check_oauth2['x5t#S256'])
    CONF.token.cache_on_issue = cache_on_issue
    CONF.token.caching = caching