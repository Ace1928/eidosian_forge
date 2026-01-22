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
def test_argparse_token(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    parser = argparse.ArgumentParser()
    c.register_argparse_arguments(parser, [])
    parser.add_argument('--os-auth-token')
    opts, _remain = parser.parse_known_args(['--os-auth-token', 'very-bad-things', '--os-auth-type', 'token'])
    cc = c.get_one(argparse=opts, validate=False)
    self.assertEqual(cc.config['auth_type'], 'token')
    self.assertEqual(cc.config['auth']['token'], 'very-bad-things')