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
def get_cinder_discovery_mock_dict(self, block_storage_version_json='block-storage-version.json', block_storage_discovery_url='https://block-storage.example.com/'):
    discovery_fixture = os.path.join(self.fixtures_directory, block_storage_version_json)
    return dict(method='GET', uri=block_storage_discovery_url, text=open(discovery_fixture, 'r').read())