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
def test_argparse_underscores(self):
    c = config.OpenStackConfig(config_files=[self.no_yaml], vendor_files=[self.no_yaml], secure_files=[self.no_yaml])
    parser = argparse.ArgumentParser()
    parser.add_argument('--os_username')
    argv = ['--os_username', 'user', '--os_password', 'pass', '--os-auth-url', 'auth-url', '--os-project-name', 'project']
    c.register_argparse_arguments(parser, argv=argv)
    opts, _remain = parser.parse_known_args(argv)
    cc = c.get_one(argparse=opts)
    self.assertEqual(cc.config['auth']['username'], 'user')
    self.assertEqual(cc.config['auth']['password'], 'pass')
    self.assertEqual(cc.config['auth']['auth_url'], 'auth-url')