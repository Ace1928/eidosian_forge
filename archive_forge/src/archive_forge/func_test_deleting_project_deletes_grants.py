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
def test_deleting_project_deletes_grants(self):
    role_path = '/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s'
    role_path = role_path % {'user_id': self.user['id'], 'project_id': self.projectA['id'], 'role_id': self.role['id']}
    self.put(role_path)
    self.delete('/projects/%(project_id)s' % {'project_id': self.projectA['id']})
    self.head(role_path, expected_status=http.client.NOT_FOUND)