import argparse
import copy
import os
from unittest import mock
import fixtures
import testtools
import yaml
from openstack import config
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
def test_set_one_cloud_updates_cloud(self):
    new_config = {'cloud': 'new_cloud', 'auth': {'password': 'newpass'}}
    resulting_cloud_config = {'auth': {'password': 'newpass', 'username': 'testuser', 'auth_url': 'http://example.com/v2'}, 'cloud': 'new_cloud', 'profile': '_test_cloud_in_our_cloud', 'region_name': 'test-region'}
    resulting_config = copy.deepcopy(base.USER_CONF)
    resulting_config['clouds']['_test-cloud_'] = resulting_cloud_config
    config.OpenStackConfig.set_one_cloud(self.cloud_yaml, '_test-cloud_', new_config)
    with open(self.cloud_yaml) as fh:
        written_config = yaml.safe_load(fh)
        written_config['cache'].pop('path', None)
        self.assertEqual(written_config, resulting_config)