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
def test_get_one_auth_defaults(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml])
    cc = c.get_one(cloud='_test-cloud_', auth={'username': 'user'})
    self.assertEqual('user', cc.auth['username'])
    self.assertEqual(defaults._defaults['auth_type'], cc.auth_type)