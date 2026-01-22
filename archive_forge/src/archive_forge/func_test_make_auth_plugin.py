import http.client
from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
from glance.common import auth
from glance.common import client
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests import utils
def test_make_auth_plugin(self):
    creds = {'strategy': 'keystone'}
    insecure = False
    with mock.patch.object(auth, 'get_plugin_from_strategy'):
        self.client.make_auth_plugin(creds, insecure)