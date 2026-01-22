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
def test_generate_temp_url_bad_time(self):
    for bad_time in ['not_an_int', -1, 1.1, '-1', '1.1', '2015-05', '2015-05-01T01:00']:
        self.assertRaisesRegex(ValueError, self.time_errmsg, self.proxy.generate_temp_url, self.url, bad_time, self.method, temp_url_key=self.key)