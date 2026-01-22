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
def test_generate_temp_url_bad_path(self):
    for bad_path in ['/v1/a/c', 'v1/a/c/o', 'blah/v1/a/c/o', '/v1//c/o', '/v1/a/c/', '/v1/a/c']:
        self.assertRaisesRegex(ValueError, self.path_errmsg, self.proxy.generate_temp_url, bad_path, 60, self.method, temp_url_key=self.key)