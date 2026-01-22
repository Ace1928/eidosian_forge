import copy
import datetime
import logging
import os
from unittest import mock
import uuid
import argparse
import configparser
import fixtures
import freezegun
import http.client
import oslo_config.fixture
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_upgradecheck import upgradecheck
from testtools import matchers
from keystone.cmd import cli
from keystone.cmd.doctor import caching
from keystone.cmd.doctor import credential
from keystone.cmd.doctor import database as doc_database
from keystone.cmd.doctor import debug
from keystone.cmd.doctor import federation
from keystone.cmd.doctor import ldap
from keystone.cmd.doctor import security_compliance
from keystone.cmd.doctor import tokens
from keystone.cmd.doctor import tokens_fernet
from keystone.cmd import status
from keystone.common import provider_api
from keystone.common.sql import upgrades
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping as identity_mapping
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.ksfixtures import policy
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import mapping_fixtures
def test_bootstrap_is_not_idempotent_when_password_does_change(self):
    self._do_test_bootstrap(self.bootstrap)
    app = self.loadapp()
    v3_password_data = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.bootstrap.username, 'password': self.bootstrap.password, 'domain': {'id': CONF.identity.default_domain_id}}}}}}
    time = datetime.datetime.utcnow()
    with freezegun.freeze_time(time) as frozen_time:
        with app.test_client() as c:
            auth_response = c.post('/v3/auth/tokens', json=v3_password_data)
            token = auth_response.headers['X-Subject-Token']
        new_passwd = uuid.uuid4().hex
        os.environ['OS_BOOTSTRAP_PASSWORD'] = new_passwd
        self._do_test_bootstrap(self.bootstrap)
        v3_password_data['auth']['identity']['password']['user']['password'] = new_passwd
        frozen_time.tick(delta=datetime.timedelta(seconds=1))
        with app.test_client() as c:
            r = c.post('/v3/auth/tokens', json=v3_password_data)
            c.get('/v3/auth/tokens', headers={'X-Auth-Token': r.headers['X-Subject-Token'], 'X-Subject-Token': token}, expected_status_code=http.client.NOT_FOUND)