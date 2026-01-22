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
def test_generate_temp_url_invalid_path(self):
    self.assertRaisesRegex(ValueError, 'path must be representable as UTF-8', self.proxy.generate_temp_url, b'/v1/a/c/\xff', self.seconds, self.method, temp_url_key=self.key)