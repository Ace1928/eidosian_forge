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
def test_get_projects_matches_federated_get_projects(self):
    ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    r = self.post('/projects', body={'project': ref})
    unauthorized_project_id = r.json['project']['id']
    r = self.get('/auth/projects', expected_status=http.client.OK)
    self.assertThat(r.json['projects'], matchers.HasLength(1))
    for project in r.json['projects']:
        self.assertNotEqual(unauthorized_project_id, project['id'])
    expected_project_id = r.json['projects'][0]['id']
    r = self.get('/OS-FEDERATION/projects', expected_status=http.client.OK)
    self.assertThat(r.json['projects'], matchers.HasLength(1))
    for project in r.json['projects']:
        self.assertEqual(expected_project_id, project['id'])