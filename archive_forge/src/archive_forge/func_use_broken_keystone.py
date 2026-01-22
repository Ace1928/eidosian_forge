import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def use_broken_keystone(self):
    self.adapter = self.useFixture(rm_fixture.Fixture())
    self.calls = []
    self._uri_registry.clear()
    self.__do_register_uris([dict(method='GET', uri='https://identity.example.com/', text=open(self.discovery_json, 'r').read()), dict(method='POST', uri='https://identity.example.com/v3/auth/tokens', status_code=400)])
    self._make_test_cloud(identity_api_version='3')