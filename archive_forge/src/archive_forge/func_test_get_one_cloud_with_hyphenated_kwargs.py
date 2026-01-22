import argparse
import copy
import os
import extras
import fixtures
import testtools
import yaml
from openstack.config import defaults
from os_client_config import cloud_config
from os_client_config import config
from os_client_config import exceptions
from os_client_config.tests import base
def test_get_one_cloud_with_hyphenated_kwargs(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    args = {'auth': {'username': 'testuser', 'password': 'testpass', 'project-id': '12345', 'auth-url': 'http://example.com/v2'}, 'region_name': 'test-region'}
    cc = c.get_one_cloud(**args)
    self.assertEqual('http://example.com/v2', cc.auth['auth_url'])