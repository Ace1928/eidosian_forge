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
def test_project_conflict_priority(self):
    """The order of priority should be
        1: env or cli settings
        2: setting from 'auth' section of clouds.yaml

        The ordering of #1 is important so that operators can use domain-wide
        inherited credentials in clouds.yaml.
        """
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cloud = {'auth_type': 'password', 'auth': {'project_id': 'my_project_id'}}
    result = c._fix_backwards_project(cloud)
    expected = {'auth_type': 'password', 'auth': {'project_id': 'my_project_id'}}
    self.assertEqual(expected, result)
    cloud = {'auth_type': 'password', 'auth': {'project_id': 'my_project_id'}, 'project_id': 'different_project_id'}
    result = c._fix_backwards_project(cloud)
    expected = {'auth_type': 'password', 'auth': {'project_id': 'different_project_id'}}
    self.assertEqual(expected, result)