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
def test_set_one_cloud_creates_file(self):
    config_dir = fixtures.TempDir()
    self.useFixture(config_dir)
    config_path = os.path.join(config_dir.path, 'clouds.yaml')
    config.OpenStackConfig.set_one_cloud(config_path, '_test_cloud_')
    self.assertTrue(os.path.isfile(config_path))
    with open(config_path) as fh:
        self.assertEqual({'clouds': {'_test_cloud_': {}}}, yaml.safe_load(fh))