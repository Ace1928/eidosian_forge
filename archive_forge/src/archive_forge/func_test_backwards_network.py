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
def test_backwards_network(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cloud = {'external_network': 'public', 'internal_network': 'private'}
    result = c._fix_backwards_networks(cloud)
    expected = {'external_network': 'public', 'internal_network': 'private', 'networks': [{'name': 'public', 'routes_externally': True, 'nat_destination': False, 'default_interface': True}, {'name': 'private', 'routes_externally': False, 'nat_destination': True, 'default_interface': False}]}
    self.assertEqual(expected, result)