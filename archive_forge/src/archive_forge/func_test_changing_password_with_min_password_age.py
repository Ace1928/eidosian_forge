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
def test_changing_password_with_min_password_age(self):
    time = datetime.datetime.utcnow()
    with freezegun.freeze_time(time) as frozen_datetime:
        new_password = uuid.uuid4().hex
        self.config_fixture.config(group='security_compliance', minimum_password_age=1)
        self.change_password(password=new_password, original_password=self.user_ref['password'], expected_status=http.client.NO_CONTENT)
        frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
        self.token = self.get_request_token(new_password, http.client.CREATED)
        self.change_password(password=uuid.uuid4().hex, original_password=new_password, expected_status=http.client.BAD_REQUEST)
        self.config_fixture.config(group='security_compliance', minimum_password_age=0)
        self.change_password(password=uuid.uuid4().hex, original_password=new_password, expected_status=http.client.NO_CONTENT)