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
def get_glance_discovery_mock_dict(self, image_version_json='image-version.json', image_discovery_url='https://image.example.com/'):
    discovery_fixture = os.path.join(self.fixtures_directory, image_version_json)
    return dict(method='GET', uri=image_discovery_url, status_code=300, text=open(discovery_fixture, 'r').read())