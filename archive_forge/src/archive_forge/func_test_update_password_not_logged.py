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
def test_update_password_not_logged(self):
    log_fix = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
    user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    self.assertNotIn(user_ref['password'], log_fix.output)
    new_password = uuid.uuid4().hex
    self.patch('/users/%s' % user_ref['id'], body={'user': {'password': new_password}})
    self.assertNotIn(new_password, log_fix.output)