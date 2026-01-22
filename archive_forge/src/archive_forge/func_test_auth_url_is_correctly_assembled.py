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
def test_auth_url_is_correctly_assembled(self):
    tuples = [('1.0', OpenStackMockHttp, {}), ('1.1', OpenStackMockHttp, {}), ('2.0', OpenStack_2_0_MockHttp, {}), ('2.0_apikey', OpenStack_2_0_MockHttp, {}), ('2.0_password', OpenStack_2_0_MockHttp, {}), ('3.x_password', OpenStackIdentity_3_0_MockHttp, {'tenant_name': 'tenant-name'}), ('3.x_appcred', OpenStackIdentity_3_0_MockHttp, {}), ('3.x_oidc_access_token', OpenStackIdentity_3_0_MockHttp, {'tenant_name': 'tenant-name'})]
    auth_urls = [('https://auth.api.example.com', ''), ('https://auth.api.example.com/', '/'), ('https://auth.api.example.com/foo/bar', '/foo/bar'), ('https://auth.api.example.com/foo/bar/', '/foo/bar/')]
    actions = {'1.0': '{url_path}/v1.0', '1.1': '{url_path}/v1.1/auth', '2.0': '{url_path}/v2.0/tokens', '2.0_apikey': '{url_path}/v2.0/tokens', '2.0_password': '{url_path}/v2.0/tokens', '3.x_password': '{url_path}/v3/auth/tokens', '3.x_appcred': '{url_path}/v3/auth/tokens', '3.x_oidc_access_token': '{url_path}/v3/OS-FEDERATION/identity_providers/user_name/protocols/tenant-name/auth'}
    user_id = OPENSTACK_PARAMS[0]
    key = OPENSTACK_PARAMS[1]
    for auth_version, mock_http_class, kwargs in tuples:
        for url, url_path in auth_urls:
            connection = self._get_mock_connection(mock_http_class=mock_http_class, auth_url=url)
            auth_url = connection.auth_url
            cls = get_class_for_auth_version(auth_version=auth_version)
            osa = cls(auth_url=auth_url, user_id=user_id, key=key, parent_conn=connection, **kwargs)
            try:
                osa = osa.authenticate()
            except Exception:
                pass
            expected_path = actions[auth_version].format(url_path=url_path).replace('//', '/')
            self.assertEqual(osa.action, expected_path)