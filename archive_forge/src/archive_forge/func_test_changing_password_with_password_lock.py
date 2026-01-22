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
def test_changing_password_with_password_lock(self):
    password = uuid.uuid4().hex
    ref = unit.new_user_ref(domain_id=self.domain_id, password=password)
    response = self.post('/users', body={'user': ref})
    user_id = response.json_body['user']['id']
    time = datetime.datetime.utcnow()
    with freezegun.freeze_time(time) as frozen_datetime:
        lock_pw_opt = options.LOCK_PASSWORD_OPT.option_name
        user_patch = {'user': {'options': {lock_pw_opt: True}}}
        self.patch('/users/%s' % user_id, body=user_patch)
        new_password = uuid.uuid4().hex
        body = {'user': {'original_password': password, 'password': new_password}}
        path = '/users/%s/password' % user_id
        self.post(path, body=body, expected_status=http.client.BAD_REQUEST)
        user_patch['user']['options'][lock_pw_opt] = False
        self.patch('/users/%s' % user_id, body=user_patch)
        path = '/users/%s/password' % user_id
        self.post(path, body=body, expected_status=http.client.NO_CONTENT)
        frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
        auth_data = self.build_authentication_request(user_id=user_id, password=new_password)
        self.v3_create_token(auth_data, expected_status=http.client.CREATED)
        path = '/users/%s' % user_id
        user = self.get(path).json_body['user']
        self.assertIn(lock_pw_opt, user['options'])
        self.assertFalse(user['options'][lock_pw_opt])
        user_patch['user']['options'][lock_pw_opt] = None
        self.patch('/users/%s' % user_id, body=user_patch)
        path = '/users/%s' % user_id
        user = self.get(path).json_body['user']
        self.assertNotIn(lock_pw_opt, user['options'])