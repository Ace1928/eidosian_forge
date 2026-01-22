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
def test_changing_expired_password_with_disabled_user_fails(self):
    self.config_fixture.config(group='security_compliance', password_expires_days=2)
    password = self._create_user_with_expired_password()
    self.user_ref['enabled'] = False
    self.patch('/users/%s' % self.user_ref['id'], body={'user': self.user_ref})
    new_password = uuid.uuid4().hex
    self.change_password(password=new_password, original_password=password, expected_status=http.client.UNAUTHORIZED)