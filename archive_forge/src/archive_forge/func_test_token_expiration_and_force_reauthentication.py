import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
def test_token_expiration_and_force_reauthentication(self):
    user_id = OPENSTACK_PARAMS[0]
    key = OPENSTACK_PARAMS[1]
    connection = self._get_mock_connection(OpenStack_2_0_MockHttp)
    auth_url = connection.auth_url
    osa = OpenStackIdentity_2_0_Connection(auth_url=auth_url, user_id=user_id, key=key, parent_conn=connection)
    mocked_auth_method = Mock(wraps=osa._authenticate_2_0_with_body)
    osa._authenticate_2_0_with_body = mocked_auth_method
    osa.auth_token = None
    osa.auth_token_expires = YESTERDAY
    count = 5
    for i in range(0, count):
        osa.authenticate(force=True)
    self.assertEqual(mocked_auth_method.call_count, count)
    osa.auth_token = None
    osa.auth_token_expires = YESTERDAY
    mocked_auth_method.call_count = 0
    self.assertEqual(mocked_auth_method.call_count, 0)
    for i in range(0, count):
        osa.authenticate(force=False)
    self.assertEqual(mocked_auth_method.call_count, 1)
    osa.auth_token = None
    mocked_auth_method.call_count = 0
    self.assertEqual(mocked_auth_method.call_count, 0)
    for i in range(0, count):
        osa.authenticate(force=False)
        if i == 0:
            osa.auth_token_expires = TOMORROW
    self.assertEqual(mocked_auth_method.call_count, 1)
    soon = datetime.datetime.utcnow() + datetime.timedelta(seconds=AUTH_TOKEN_EXPIRES_GRACE_SECONDS - 1)
    osa.auth_token = None
    mocked_auth_method.call_count = 0
    self.assertEqual(mocked_auth_method.call_count, 0)
    for i in range(0, count):
        if i == 0:
            osa.auth_token_expires = soon
        osa.authenticate(force=False)
    self.assertEqual(mocked_auth_method.call_count, 1)