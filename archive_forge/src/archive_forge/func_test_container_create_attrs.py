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
def test_container_create_attrs(self):
    self.verify_create(self.proxy.create_container, container.Container, method_args=['container_name'], expected_args=[], expected_kwargs={'name': 'container_name', 'x': 1, 'y': 2, 'z': 3})