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
def get_keystone_v3_token(self, project_name='admin'):
    return dict(method='POST', uri='https://identity.example.com/v3/auth/tokens', headers={'X-Subject-Token': self.getUniqueString('KeystoneToken')}, json=self.os_fixture.v3_token, validate=dict(json={'auth': {'identity': {'methods': ['password'], 'password': {'user': {'domain': {'name': 'default'}, 'name': 'admin', 'password': 'password'}}}, 'scope': {'project': {'domain': {'name': 'default'}, 'name': project_name}}}}))