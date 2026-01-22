from hashlib import sha1
import random
import string
import tempfile
import time
from unittest import mock
import requests_mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.object_store.v1 import account
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
from openstack.tests.unit import test_proxy_base
@mock.patch('hmac.HMAC')
@mock.patch('time.time', return_value=1400000000)
def test_generate_temp_url_ip_range(self, time_mock, hmac_mock):
    hmac_mock().hexdigest.return_value = 'temp_url_signature'
    ip_ranges = ['1.2.3.4', '1.2.3.4/24', '2001:db8::', b'1.2.3.4', b'1.2.3.4/24', b'2001:db8::']
    path = '/v1/AUTH_account/c/o/'
    expected_url = path + '?temp_url_sig=temp_url_signature&temp_url_expires=1400003600&temp_url_ip_range='
    for ip_range in ip_ranges:
        hmac_mock.reset_mock()
        url = self.proxy.generate_temp_url(path, self.seconds, self.method, temp_url_key=self.key, ip_range=ip_range)
        key = self.key
        if not isinstance(key, bytes):
            key = key.encode('utf-8')
        if isinstance(ip_range, bytes):
            ip_range_expected_url = expected_url + ip_range.decode('utf-8')
            expected_body = '\n'.join(['ip=' + ip_range.decode('utf-8'), self.method, '1400003600', path]).encode('utf-8')
        else:
            ip_range_expected_url = expected_url + ip_range
            expected_body = '\n'.join(['ip=' + ip_range, self.method, '1400003600', path]).encode('utf-8')
        self.assertEqual(url, ip_range_expected_url)
        self.assertEqual(hmac_mock.mock_calls, [mock.call(key, expected_body, sha1), mock.call().hexdigest()])
        self.assertIsInstance(url, type(path))